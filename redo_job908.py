"""Replace reviewed automatic rectangles in Job 908 with semantic masks.

Offline by default. --apply uses a local Terminal login, verifies all original
pixels and annotations, creates and verifies masks, then removes only the old
automatic rectangle IDs. No annotation import, ground-truth lookup, or task edits.
"""

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from cvat_sdk.masks import decode_mask, encode_mask
from cvat_sdk.attributes import attribute_value_validator
from cvat_sdk.api_client import models

from annotation_writer import verify_create_schema, verify_patch_schema
from auth import DiagnosticError, Redactor, read_credentials
from config import Config
from cvat_client import CvatClient
from frame_mapping import JobScope, build_frames, validate_job
from prepare_job908_semantic import INPUT, OUTPUT, NAMES


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_digest(image):
    return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()


def annotation_digest(data):
    return digest({kind: sorted(data.get(kind, []), key=lambda obj: obj["id"])
                   for kind in ("shapes", "tracks", "tags", "intervals")})


def semantic_labels(labels):
    result = {}
    for name in NAMES:
        matches = [label for label in labels if label["name"] == name]
        if len(matches) != 1 or matches[0].get("type", "any") not in ("any", "mask"):
            raise ValueError(f"Cần đúng một label cho phép mask: {name}.")
        result[name] = matches[0]
    ids = [label.get("id") for label in result.values()]
    if any(type(item) is not int or item <= 0 for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("ID của 19 label semantic không hợp lệ hoặc trùng nhau.")
    return result


def semantic_label_contract(labels):
    """Only the schema used by the prepared masks; ignore API/display metadata."""
    targets = semantic_labels(labels)
    result = []
    for name in sorted(targets):
        label = targets[name]
        attributes = []
        for attribute in label.get("attributes", []):
            attributes.append({"id": attribute["id"], "name": attribute.get("name"),
                               "input_type": attribute.get("input_type"),
                               "mutable": attribute.get("mutable", False),
                               "default_value": attribute.get("default_value"),
                               "values": (list(attribute.get("values", [])) if attribute.get("input_type") == "number"
                                          else sorted(attribute.get("values", [])))})
        result.append({"id": label["id"], "name": name, "geometry": "mask",
                       "parent_id": label.get("parent_id"),
                       "attributes": sorted(attributes, key=lambda item: item["id"])})
    return result


def attribute_defaults(label):
    """Use API-provided defaults and validate them with CVAT's own validator."""
    defaults = {}
    names = set()
    for attribute in label.get("attributes", []):
        spec_id = attribute.get("id")
        name = attribute.get("name")
        if type(spec_id) is not int or spec_id <= 0 or spec_id in defaults or name in names:
            raise ValueError(f"Thuộc tính của label {label['name']} bị trùng hoặc sai ID.")
        value = attribute.get("default_value")
        if value is None:
            raise ValueError(f"Thuộc tính {label['name']}/{name} chưa có giá trị mặc định.")
        value = str(value)
        try:
            spec = models.AttributeRequest(name=name, input_type=attribute["input_type"],
                                           mutable=attribute.get("mutable", False),
                                           values=attribute.get("values", []), default_value=value)
            valid = attribute_value_validator(spec)(value)
        except Exception:
            valid = False
        if not valid:
            raise ValueError(f"Giá trị mặc định của {label['name']}/{name} không hợp lệ theo CVAT.")
        defaults[spec_id] = value
        names.add(name)
    return defaults


def bind_shape_attributes(shape, labels):
    """Fill new default fields; preserve every already specified value."""
    label = labels[shape["label_id"]]
    values = attribute_defaults(label)
    supplied = shape.get("attributes", [])
    ids = [item.get("spec_id") for item in supplied]
    if len(ids) != len(set(ids)) or not set(ids) <= set(values):
        raise ValueError(f"Thuộc tính của annotation label {label['name']} không thuộc schema hiện tại.")
    values.update({item["spec_id"]: item["value"] for item in supplied})
    return {**shape, "attributes": [{"spec_id": key, "value": value} for key, value in sorted(values.items())]}


def compatible_attribute_additions(expected, actual):
    """Accept added fields with valid defaults, never changes to reviewed fields."""
    before = {a["id"]: a for a in expected}
    after = {a["id"]: a for a in actual}
    if len(before) != len(expected) or len(after) != len(actual):
        return False
    return all(after.get(key) == value for key, value in before.items())


def verify_live_labels(plan, labels):
    expected = plan.get("semantic_label_contract")
    if expected is None:
        # v1 plans only stored a raw API digest. Authenticate the cached baseline
        # against that digest before deriving its meaningful semantic schema.
        baseline = read_json(INPUT / "labels.json")
        if digest(baseline) != plan["labels_digest"]:
            raise ValueError("Bản sao labels khác bản đã review; không thể xác minh schema gốc.")
        expected = semantic_label_contract(baseline)
    report = {"job_id": 908, "plan_id": plan["plan_id"], "expected": expected,
              "compatible": False, "raw_metadata_changed": digest(labels) != plan["labels_digest"]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels_live.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        actual = semantic_label_contract(labels)
        before = {item["name"]: item for item in expected}
        after = {item["name"]: item for item in actual}
        differences, additions = [], []
        live_by_name = semantic_labels(labels)
        for name in before:
            fields = [key for key in before[name] if before[name][key] != after[name][key]]
            if "attributes" in fields and compatible_attribute_additions(before[name]["attributes"], after[name]["attributes"]):
                attribute_defaults(live_by_name[name])
                old_ids = {item["id"] for item in before[name]["attributes"]}
                additions.append({"name": name, "attributes": [a for a in after[name]["attributes"] if a["id"] not in old_ids]})
                fields.remove("attributes")
            if fields:
                differences.append({"name": name, "fields": fields})
        report.update(actual=actual, differences=differences, attribute_additions=additions, compatible=not differences)
    except ValueError as error:
        report.update(error=str(error))
        (OUTPUT / "label_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError(f"Schema semantic chưa hợp lệ: {error} Xem outputs/job908_semantic/label_comparison.json.") from None
    (OUTPUT / "label_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if differences:
        details = "; ".join(f"{row['name']} ({', '.join(row['fields'])})" for row in differences)
        raise ValueError(f"Schema semantic thay đổi: {details}. Xem outputs/job908_semantic/label_comparison.json.")
    if report["raw_metadata_changed"]:
        print("[OK] 19 label semantic khớp ID/tên/geometry; metadata và thuộc tính mới có default hợp lệ đã được đối chiếu.")
    if additions:
        print("[OK] Bổ sung thuộc tính mặc định hiện tại cho: " + ", ".join(row["name"] for row in additions))
    return True


def make_scope(job, metadata, labels):
    targets = semantic_labels(labels)
    return JobScope(908, build_frames(job, metadata), {label["id"]: label for label in labels},
                    allowed_shape_types={"mask"},
                    label_geometry={label["id"]: "mask" for label in targets.values()})


def validate_originals(original, frames):
    if any(original.get(kind) for kind in ("tracks", "tags", "intervals")):
        raise ValueError("Có track/tag/interval cần review riêng trước khi thay nhãn.")
    shapes = original.get("shapes", [])
    ids = [shape.get("id") for shape in shapes]
    if len(ids) != len(set(ids)) or any(type(item) is not int for item in ids):
        raise ValueError("ID nhãn cũ không hợp lệ.")
    if any(s.get("type") != "rectangle" or s.get("source") != "auto" or s.get("frame") not in frames
           for s in shapes):
        raise ValueError("Chỉ thay các box tự động trong bản đã review; không tự xóa nhãn vẽ tay.")


def build_plan():
    job, metadata, labels = (read_json(INPUT / f"{name}.json") for name in ("job", "metadata", "labels"))
    if (job["id"], job["task_id"], job["organization"]) != (908, 211, 1):
        raise ValueError("Bản đọc không thuộc Task 211 / Job 908 / cohort-5a.")
    scope = make_scope(job, metadata, labels)
    original = read_json(INPUT / "annotations_before.json")
    validate_originals(original, scope.frames)
    reviews = read_json(OUTPUT / "visual_review.json")
    targets = semantic_labels(labels)
    shapes, frames = [], []
    for number, info in sorted(scope.frames.items()):
        source = INPUT / f"frame_{number:06d}.png"
        classes = OUTPUT / f"classes_{number:06d}.png"
        corrections = OUTPUT / f"corrections_{number:06d}.json"
        review = reviews.get(str(number), {})
        if review.get("passes", 0) < 2:
            raise ValueError(f"Frame {number} chưa được review hai lượt.")
        hashes = {"source_sha256": file_digest(source), "classes_sha256": file_digest(classes),
                  "corrections_sha256": file_digest(corrections)}
        if any(review.get(key) != value for key, value in hashes.items()):
            raise ValueError(f"Frame {number} thay đổi sau khi review.")
        with Image.open(source) as image:
            if image.size != (info.width, info.height):
                raise ValueError("Kích thước ảnh nguồn sai.")
            pixels = pixel_digest(image)
        with Image.open(classes) as image:
            category = np.asarray(image)
        if category.shape != (info.height, info.width) or not np.isin(category, [*range(19), 255]).all():
            raise ValueError("Bản đồ class không đúng kích thước hoặc taxonomy.")
        if not (category != 255).any():
            raise ValueError("Không chấp nhận toàn ảnh rỗng.")
        for class_id, name in enumerate(NAMES):
            bitmap = category == class_id
            if not bitmap.any():
                continue
            label = targets[name]
            shape = {"frame": number, "label_id": label["id"], "type": "mask", "source": "auto",
                     "points": encode_mask(bitmap), "rotation": 0., "occluded": False, "z_order": 0,
                     "attributes": [{"spec_id": a["id"], "value": str(a["default_value"])}
                                    for a in label.get("attributes", [])]}
            scope.validate_shape(shape)
            if not np.array_equal(bitmap, decode_mask(shape["points"], image_width=info.width, image_height=info.height)):
                raise ValueError("Mask RLE không giữ nguyên pixel.")
            shapes.append(shape)
        frames.append({"frame": number, "width": info.width, "height": info.height,
                       "pixels_sha256": pixels, **hashes, "void_pixels": int((category == 255).sum())})
    plan = {"format": "job908-reviewed-semantic-v1", "job_id": 908, "task_id": 211,
            "organization": "cohort-5a", "original_digest": annotation_digest(original),
            "original_ids": [s["id"] for s in original["shapes"]], "labels_digest": digest(labels),
            "semantic_label_contract": semantic_label_contract(labels),
            "frames": frames, "shapes": shapes, "review_notes": reviews.get("issues", []),
            "state_after_save": "unchanged", "requires_reviewer": bool(reviews.get("issues"))}
    plan["plan_id"] = digest(plan)
    (OUTPUT / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    return plan


def load_plan():
    plan = read_json(OUTPUT / "plan.json")
    if (plan.get("job_id"), plan.get("task_id"), plan.get("organization"), plan.get("format")) != (
            908, 211, "cohort-5a", "job908-reviewed-semantic-v1"):
        raise ValueError("Phạm vi bản sửa không hợp lệ.")
    if plan.get("plan_id") != digest({key: value for key, value in plan.items() if key != "plan_id"}):
        raise ValueError("Nội dung bản sửa đã thay đổi.")
    for frame in plan["frames"]:
        n = frame["frame"]
        for key, path in (("source_sha256", INPUT / f"frame_{n:06d}.png"),
                          ("classes_sha256", OUTPUT / f"classes_{n:06d}.png"),
                          ("corrections_sha256", OUTPUT / f"corrections_{n:06d}.json")):
            if file_digest(path) != frame[key]:
                raise ValueError(f"Frame {n} khác bản đã chuẩn bị; cần review lại.")
    return plan


def mask_signature(shape, scope):
    if shape.get("type") != "mask":
        raise ValueError("Nhãn mới không phải mask.")
    # Strip API-generated fields, and validate the complete new-shape semantics.
    candidate = {key: shape[key] for key in ("frame", "label_id", "type", "source", "points", "attributes",
                                           "occluded", "rotation", "z_order") if key in shape}
    candidate = bind_shape_attributes(candidate, scope.labels)
    scope.validate_shape(candidate)
    if candidate.get("occluded", False):
        raise ValueError("Semantic mask chỉ tô phần nhìn thấy, không bật occluded.")
    info = scope.frames[shape["frame"]]
    pixels = decode_mask(shape["points"], image_width=info.width, image_height=info.height)
    return shape["frame"], shape["label_id"], hashlib.sha256(pixels.tobytes()).hexdigest()


def classify_current(current, original, plan, scope):
    """Recognize only the exact original, original+all masks, or final mask set."""
    if any(current.get(kind) for kind in ("tags", "tracks", "intervals")):
        raise ValueError("Annotation hiện tại có dữ liệu mới ngoài bản đã review; dừng.")
    old = {s["id"]: s for s in original["shapes"]}
    current_old = [s for s in current.get("shapes", []) if s.get("id") in old]
    fresh = [s for s in current.get("shapes", []) if s.get("id") not in old]
    if current_old and ({s["id"] for s in current_old} != set(old) or
                        any(bind_shape_attributes(s, scope.labels) != bind_shape_attributes(old[s["id"]], scope.labels)
                            for s in current_old)):
        raise ValueError("Box cũ đã bị sửa hoặc xóa một phần; dừng trước khi ghi.")
    if not fresh:
        if annotation_digest(original) != plan["original_digest"] or len(current_old) != len(old):
            raise ValueError("Annotation hiện tại khác bản sao; cần đối chiếu lại.")
        return "original", []
    expected = Counter(mask_signature(s, scope) for s in plan["shapes"])
    actual = Counter(mask_signature(s, scope) for s in fresh)
    ids = [s.get("id") for s in fresh]
    if (actual != expected or len(ids) != len(set(ids)) or any(type(i) is not int for i in ids)):
        raise ValueError("Nhãn hiện tại có thay đổi khác/ghi chưa đủ; giữ box cũ, không tự gửi lại.")
    return ("created" if current_old else "finished"), fresh


def verify_delete_schema(schema):
    verify_create_schema(schema, required_types={"mask"}, required_sources={"auto"})
    verify_patch_schema(schema, action="update", required_types={"mask"})

    def resolve(obj):
        while "$ref" in obj:
            node = schema
            for token in obj["$ref"][2:].split("/"):
                node = node[token]
            obj = node
        return obj

    for path in ("/api/jobs/{id}/annotations/", "/api/jobs/{id}/annotations"):
        if path in schema.get("paths", {}):
            operation = schema["paths"][path]["patch"]
            for item in operation.get("parameters", []):
                parameter = resolve(item)
                if parameter.get("name") == "action" and "delete" in resolve(parameter.get("schema", {})).get("enum", []):
                    return True
    raise ValueError("Schema không xác nhận xóa riêng các shape ID.")


def record_status(phase, plan, **details):
    status = {"phase": phase, "plan_id": plan["plan_id"], "job_id": 908, **details}
    (OUTPUT / "apply_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[JOB 908]", phase, flush=True)


@contextmanager
def exclusive_apply():
    """OS-managed lock; released on process exit, including an interrupted login."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "apply.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("Một phiên gán Job 908 đang chạy. Dùng Terminal đó hoặc dừng phiên cũ bằng Ctrl+C trước.") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def record_stop(plan, **details):
    if plan is None:
        return
    try:
        previous = read_json(OUTPUT / "apply_status.json")
        record_status("stopped", plan, previous_phase=previous.get("phase"),
                      verification_required=True, **details)
    except (OSError, ValueError):
        print("Không lưu được trạng thái dừng; chưa xác nhận kết quả ghi.")


def apply_plan(client, user, plan):
    original = read_json(INPUT / "annotations_before.json")
    if annotation_digest(original) != plan["original_digest"]:
        raise ValueError("Bản sao nhãn cũ khác bản sửa.")

    def refresh_scope():
        job = client.job()
        organization = client.organization(job["organization"])
        validate_job(client.config, job, organization)
        if (job.get("assignee") or {}).get("id") != user["id"]:
            raise ValueError("Job không được giao cho tài khoản đang đăng nhập.")
        if job.get("stage") != "annotation" or job.get("parent_job_id") or job.get("replicas_count", 0):
            raise ValueError("Job không ở giai đoạn annotation thông thường; dừng.")
        labels = client.job_labels()
        verify_live_labels(plan, labels)
        scope = make_scope(job, client.metadata(), labels)
        expected_frames = {f["frame"]: (f["width"], f["height"]) for f in plan["frames"]}
        if {n: (f.width, f.height) for n, f in scope.frames.items()} != expected_frames:
            raise ValueError("Phạm vi frame thay đổi; dừng.")
        return scope

    scope = refresh_scope()
    execution_plan = {**plan, "shapes": [bind_shape_attributes(shape, scope.labels) for shape in plan["shapes"]]}
    for shape in execution_plan["shapes"]:
        scope.validate_shape(shape)
    validate_originals(original, scope.frames)
    verify_delete_schema(client.schema())
    for index, frame in enumerate(plan["frames"], 1):
        raw = client.frame_bytes(frame["frame"], scope.frames)
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (frame["width"], frame["height"]) or pixel_digest(image) != frame["pixels_sha256"]:
                raise ValueError("Ảnh trên CVAT khác ảnh đã review; không ghi nhãn.")
        record_status("verifying_frames", plan, checked=index, total=len(plan["frames"]))
    current = client.annotations()
    state, fresh = classify_current(current, original, execution_plan, scope)
    backup = OUTPUT / f"before_apply_{plan['plan_id'][:12]}.json"
    if not backup.exists():
        backup.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    if state == "original":
        # One request; if its result is uncertain, the next run reads and reconciles.
        record_status("creating_masks", plan, count=len(plan["shapes"]))
        client.create_shapes(execution_plan["shapes"], scope=scope, schema_verified=True, confirmed=True)
        current = client.annotations()
        state, fresh = classify_current(current, original, execution_plan, scope)
        if state != "created":
            raise ValueError("Không xác minh được đủ mask mới; giữ nguyên box cũ.")
    if state == "created":
        record_status("masks_verified", plan, created_ids=[s["id"] for s in fresh])
        scope = refresh_scope()
        current = client.annotations()
        state, fresh = classify_current(current, original, execution_plan, scope)
        if state != "created":
            raise ValueError("Dữ liệu thay đổi trước khi xóa box cũ; dừng.")
        # Send the current originals, preserving any API-materialized defaults.
        old_ids = set(plan["original_ids"])
        body = {"shapes": [s for s in current["shapes"] if s["id"] in old_ids],
                "tags": [], "tracks": [], "intervals": []}
        record_status("removing_old_rectangles", plan, count=len(original["shapes"]))
        client._call("Xóa riêng box tự động cũ đã sao lưu", None, "PATCH", "/api/jobs/908/annotations/",
                     params={"action": "delete"}, json=body)
        state, fresh = classify_current(client.annotations(), original, execution_plan, scope)
        if state != "finished":
            raise ValueError("Chưa xác minh xong bước thay box; không tự lặp yêu cầu xóa.")
    record_status("annotations_verified", plan, masks=len(fresh), frames=len(plan["frames"]),
                  removed_old_ids=plan["original_ids"], review_notes=plan["review_notes"], state_changed=False)
    return fresh


def post_review_issues(client, plan):
    """Legacy explicit helper. Normal apply no longer creates review Issues."""
    comments = client._list("Ghi chú review của Job 908", None, "/api/comments", {"job_id": 908})
    messages = {item.get("message") for item in comments}
    created = []
    for note in plan["review_notes"]:
        message = f"[Job908 semantic {plan['plan_id'][:12]}] {note['type']}: {note['message']}"
        if message in messages:
            continue
        payload = {"job": 908, "frame": note["frame"], "position": note["position"],
                   "message": message, "resolved": False}
        result = client._call("Tạo Issue review theo frame", None, "POST", "/api/issues", json=payload)
        if result.get("job") != 908 or result.get("frame") != note["frame"] or not result.get("id"):
            raise ValueError("Phản hồi tạo Issue chưa xác minh được; không tự gửi lại.")
        messages.add(message)
        created.append(result["id"])
    return created


def remove_auto_issues(client, user, plan):
    """Delete only this plan's generated Issues; preserve masks and manual Issues."""
    if not plan.get("review_notes"):
        return []
    job = client.job()
    validate_job(client.config, job, client.organization(job["organization"]))
    if (job.get("assignee") or {}).get("id") != user["id"]:
        raise ValueError("Job 908 không được giao cho tài khoản đang đăng nhập.")
    operation = client.schema().get("paths", {}).get("/api/issues/{id}", {}).get("delete", {})
    if operation.get("operationId") != "issues_destroy":
        raise ValueError("Schema không xác nhận endpoint xóa Issue.")
    expected = {f"[Job908 semantic {plan['plan_id'][:12]}] {note['type']}: {note['message']}": note
                for note in plan["review_notes"]}
    issues = client._list("Issues của Job 908", None, "/api/issues", {"job_id":908})
    comments = client._list("Comments của Job 908", None, "/api/comments", {"job_id":908})
    marked = {item["issue"]: expected[item["message"]] for item in comments
              if item.get("message") in expected and type(item.get("issue")) is int}
    targets = [issue for issue in issues if issue.get("id") in marked and issue.get("job") == 908
               and issue.get("frame") == marked[issue["id"]]["frame"]
               and (issue.get("owner") or {}).get("id") == user["id"]]
    if not targets:
        print("[OK] Không còn Issue tự động của bản sửa này.")
        return []
    before = client.annotations()
    target_ids = {issue["id"] for issue in targets}
    backup = OUTPUT / f"issues_before_removal_{plan['plan_id'][:12]}.json"
    if not backup.exists():
        backup.write_text(json.dumps({"job_id":908,"issues":targets,
                                     "comments":[c for c in comments if c.get("issue") in target_ids]},
                                    ensure_ascii=False,indent=2),encoding="utf-8")
    removed = []
    for issue in targets:
        item = client._call("Kiểm tra Issue tự động", None, "GET", f"/api/issues/{issue['id']}")
        if (item.get("id"),item.get("job"),item.get("frame"),(item.get("owner") or {}).get("id")) != (
                issue["id"],908,issue["frame"],user["id"]):
            raise ValueError("Issue thay đổi phạm vi/người tạo; dừng trước khi xóa.")
        # DELETE returns HTTP 204 without a JSON body.
        client._call("Xóa Issue tự động",None,"DELETE",f"/api/issues/{issue['id']}",binary=True)
        removed.append(issue["id"])
        print(f"[OK] Đã xóa vùng gạch đỏ {len(removed)}/{len(targets)}",flush=True)
    remaining = client._list("Kiểm tra Issues còn lại",None,"/api/issues",{"job_id":908})
    if target_ids & {issue["id"] for issue in remaining}:
        raise ValueError("Chưa xác minh xóa hết Issue tự động; chạy lại để đọc trạng thái hiện tại.")
    if annotation_digest(before) != annotation_digest(client.annotations()):
        raise ValueError("Annotation thay đổi trong lúc dọn Issue. Công cụ chỉ gửi DELETE Issue; cần đọc lại Job.")
    (OUTPUT / "issue_cleanup_status.json").write_text(json.dumps(
        {"phase":"issues_removed","job_id":908,"plan_id":plan["plan_id"],
         "removed_issue_ids":removed,"annotations_unchanged":True},ensure_ascii=False,indent=2),encoding="utf-8")
    return removed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Chuẩn bị bản thay nhãn từ các mask đã review hai lượt.")
    parser.add_argument("--apply", action="store_true", help="Đăng nhập Terminal và thay box cũ bằng bản mask đã chuẩn bị.")
    parser.add_argument("--auth", choices=("auto", "password", "pat", "session"), default="password")
    parser.add_argument("--frame-timeout", type=float, default=180)
    parser.add_argument("--audit-labels", action="store_true", help="Đăng nhập và chỉ kiểm tra 19 label semantic, không tải ảnh hoặc ghi.")
    parser.add_argument("--remove-auto-issues", action="store_true", help="Chỉ xóa vùng gạch đỏ do Issue tự động tạo; giữ nguyên nhãn mask.")
    args = parser.parse_args(argv)
    if args.audit_labels and (args.apply or args.prepare):
        parser.error("--audit-labels chạy riêng, không kết hợp --apply/--prepare.")
    if args.remove_auto_issues and (args.apply or args.prepare or args.audit_labels):
        parser.error("--remove-auto-issues chạy riêng.")
    client = None
    redactor = Redactor()
    plan = None
    active_plan = None
    locks = ExitStack()
    try:
        if args.remove_auto_issues:
            plan = load_plan()
            print("Job 908: chỉ xóa các Issue tự động gây gạch đỏ; giữ nguyên nhãn mask.")
        elif not args.audit_labels:
            plan = build_plan() if args.prepare else load_plan()
            print(f"Task 211 / Job 908 / cohort-5a: {len(plan['frames'])} frame; "
                  f"thay {len(plan['original_ids'])} box tự động bằng {len(plan['shapes'])} mask semantic.")
            print(f"Bản sửa: {plan['plan_id'][:12]}; còn {len(plan['review_notes'])} vùng cần reviewer kiểm tra.")
            print("Không đổi Stage/State hoặc cấu hình labels; mật khẩu chỉ nhập tại Terminal.")
            if not args.apply:
                print("DRY RUN: chưa đăng nhập hoặc ghi CVAT.")
                print(r"Để lưu, chạy trong Terminal: .\.venv\Scripts\python.exe .\annotate_job908.py --apply")
                return 0
        if args.apply or args.remove_auto_issues:
            locks.enter_context(exclusive_apply())
            if args.apply:
                active_plan = plan
        print("Nhập username và mật khẩu ẩn tại Terminal này. Không gửi mật khẩu vào chat.", flush=True)
        if plan and not args.remove_auto_issues:
            record_status("awaiting_login", plan)
        credentials = read_credentials(args.auth, None, redactor)
        config = Config(task_id=211, job_id=908, backend="rest", frame_timeout=args.frame_timeout)
        client = CvatClient(config, redactor)
        client.login(credentials)
        credentials = None
        user = client.current_user()
        client.set_organization()
        if args.remove_auto_issues:
            removed = remove_auto_issues(client,user,plan)
            print(f"[OK] Đã xóa {len(removed)} Issue tự động. Tải lại trang Job 908 để bỏ gạch đỏ.")
            return 0
        if args.audit_labels:
            job = client.job()
            validate_job(config, job, client.organization(job["organization"]))
            targets = semantic_labels(client.job_labels())
            rows = [{"id": label["id"], "name": name, "type": label.get("type", "any"),
                     "output_geometry": "mask"} for name, label in targets.items()]
            for row in rows:
                print(f"ID {row['id']} | {row['name']} | type={row['type']} | output=mask")
            OUTPUT.mkdir(parents=True, exist_ok=True)
            (OUTPUT / "semantic_label_audit.json").write_text(
                json.dumps({"job_id":908,"task_id":211,"labels":rows,"read_only":True},ensure_ascii=False,indent=2),encoding="utf-8")
            print("[OK] Đủ 19 label cho mask semantic. Không tải ảnh hoặc ghi annotation.")
            return 0
        apply_plan(client, user, plan)
        removed = remove_auto_issues(client,user,plan)
        record_status("saved", plan, masks=len(plan["shapes"]),
                      frames=len(plan["frames"]), removed_auto_issue_ids=removed, state_changed=False)
        print("[OK] Đã lưu và xác minh mask. Ghi chú review giữ trong outputs/job908_semantic/README_REVIEW.md.")
        print("https://cvat.edufun.ai/tasks/211/jobs/908")
        return 0
    except DiagnosticError as error:
        print(f"[DỪNG] {error.step} | HTTP {error.status or 'N/A'} | {redactor.clean(error.detail)}")
        record_stop(active_plan, step=error.step, http_status=error.status, detail=redactor.clean(error.detail))
        print("Không tự gửi lại yêu cầu ghi. Lần chạy sau sẽ đọc lại để đối chiếu trước.")
        return 1
    except (ValueError, OSError, EOFError, KeyboardInterrupt) as error:
        print("[DỪNG]", redactor.clean(error) or "Đã hủy.")
        record_stop(active_plan, detail=redactor.clean(error) or "Đã hủy.")
        return 2
    except Exception as error:
        print(f"[DỪNG] {type(error).__name__}; không in nội dung có thể chứa thông tin đăng nhập.")
        record_stop(active_plan, error_type=type(error).__name__)
        return 1
    finally:
        if client:
            client.close()
        locks.close()


if __name__ == "__main__":
    raise SystemExit(main())
