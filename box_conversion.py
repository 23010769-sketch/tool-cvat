"""Opt-in conversion of generated shapes to rectangles, preserving existing boxes.

No frames/models/dataset export/import. A preview and exact confirmation are
required for each stage. Extra component boxes are created before changing the
original mask, so a failed create cannot discard its original geometry.
"""

import copy
import math
import sys

from annotation_writer import (annotation_snapshot, apply_shapes, verify_create_schema,
                               verify_patch_schema, verify_preserved)
from bdd_reference import enclosing_box
from frame_mapping import build_frames, validate_job


def geometry_boxes(shape, info, min_pixels):
    if shape.get("rotation", 0) != 0 or shape.get("elements"):
        raise ValueError("Chưa hỗ trợ chuyển shape xoay/skeleton; dừng trước khi ghi.")
    points = shape.get("points", [])
    if not all(type(p) in (int, float) and math.isfinite(p) for p in points):
        raise ValueError("Geometry cũ có tọa độ không hợp lệ.")
    if shape["type"] == "mask":
        import cv2
        import numpy as np
        from cvat_sdk.masks import decode_mask
        if len(points) < 5 or not all(p == int(p) for p in points):
            raise ValueError("RLE mask cũ không hợp lệ.")
        *runs, x1, y1, x2, y2 = (int(p) for p in points)
        if (not 0 <= x1 <= x2 < info.width or not 0 <= y1 <= y2 < info.height
                or any(n < 0 for n in runs) or sum(runs) != (x2 - x1 + 1) * (y2 - y1 + 1)):
            raise ValueError("RLE/bbox mask cũ không khớp kích thước ảnh.")
        bitmap = decode_mask(points, image_width=info.width, image_height=info.height)
        count, _, stats, _ = cv2.connectedComponentsWithStats(bitmap.astype(np.uint8), connectivity=8)
        components = sorted((tuple(int(v) for v in stats[k]) for k in range(1, count)), key=lambda row: -row[4])
        if not components:
            raise ValueError("Mask cũ rỗng; cần kiểm tra trước khi đổi định dạng.")
        selected = [row for row in components if row[4] >= min_pixels] or components[:1]
        return [(float(x), float(y), float(x + w), float(y + h)) for x, y, w, h, _ in selected]
    if shape["type"] in ("polygon", "polyline"):
        minimum = 6 if shape["type"] == "polygon" else 4
        if len(points) < minimum or len(points) % 2:
            raise ValueError("Polygon/polyline cũ không hợp lệ.")
        return [enclosing_box(list(zip(points[::2], points[1::2])), info.width, info.height)]
    raise ValueError("Chưa hỗ trợ chuyển geometry này thành box.")


def validate_update(updated, original, scope):
    if (not original or type(updated.get("id")) is not int or updated["id"] != original.get("id")
            or original.get("source") not in ("auto", "file")
            or original.get("type") not in ("mask", "polygon", "polyline")
            or updated.get("frame") not in scope.frames or updated.get("label_id") not in scope.labels):
        raise ValueError("Update không thuộc shape tự động đã xem trước trong Job.")
    unchanged = set(original) | set(updated)
    if any(original.get(key) != updated.get(key) for key in unchanged - {"type", "points", "rotation"}):
        raise ValueError("Chuyển box không được đổi ID/frame/label/thuộc tính của shape cũ.")
    if updated.get("type") != "rectangle" or updated.get("rotation") != 0:
        raise ValueError("Chuyển định dạng chỉ tạo rectangle, rotation=0.")
    label = scope.labels[updated["label_id"]]
    if label.get("type", "any") not in ("any", "rectangle"):
        raise ValueError("Label không cho phép rectangle; dừng, không sửa label schema.")
    points = updated.get("points", [])
    if len(points) != 4 or not all(type(p) in (int, float) and math.isfinite(p) for p in points):
        raise ValueError("Box chuyển đổi không hợp lệ.")
    info = scope.frames[updated["frame"]]
    x1, y1, x2, y2 = points
    if not (0 <= x1 < x2 <= info.width and 0 <= y1 < y2 <= info.height):
        raise ValueError("Box chuyển đổi nằm ngoài ảnh hoặc không có diện tích.")


def plan_conversions(annotations, scope, numbers, min_pixels):
    updates, additions, originals, skipped = [], [], {}, 0
    for track in annotations.get("tracks", []):
        if track.get("source") in ("auto", "file") and any(s.get("type") != "rectangle" for s in track.get("shapes", [])):
            raise ValueError("Job có track không phải rectangle; chưa hỗ trợ chuyển track tự động.")
    for old in annotations.get("shapes", []):
        if old.get("frame") not in numbers or old.get("type") == "rectangle":
            continue
        if old.get("source") not in ("auto", "file"):
            skipped += 1
            continue  # Do not change manually authored annotation.
        if old.get("label_id") not in scope.labels or type(old.get("id")) is not int:
            raise ValueError("Shape cũ có ID/label không thuộc phạm vi Job.")
        boxes = geometry_boxes(old, scope.frames[old["frame"]], min_pixels)
        update = {**copy.deepcopy(old), "type": "rectangle", "points": list(boxes[0]), "rotation": 0.}
        validate_update(update, old, scope)
        updates.append(update)
        originals[old["id"]] = copy.deepcopy(old)
        for points in boxes[1:]:
            if old.get("group", 0):
                raise ValueError("Mask có group và nhiều vùng rời; cần kiểm tra trước khi tách box.")
            addition = {key: copy.deepcopy(value) for key, value in update.items()
                        if key in {"frame", "label_id", "type", "points", "source", "attributes", "occluded", "rotation", "z_order", "score"}}
            addition["points"] = list(points)
            scope.validate_shape(addition)
            additions.append(addition)
    return updates, additions, originals, skipped


def revalidate_scope(client, scope):
    job = client.job()
    organization = client.organization(job["organization"]) if client.config.organization else None
    validate_job(client.config, job, organization)
    if job.get("parent_job_id") or job.get("replicas_count", 0):
        raise ValueError("Chưa hỗ trợ chuyển geometry của Job consensus.")
    if build_frames(job, client.metadata()) != scope.frames:
        raise ValueError("Metadata đã đổi sau xem trước; dừng chuyển box.")
    if {label["id"]: label for label in client.job_labels()} != scope.labels:
        raise ValueError("Labels đã đổi sau xem trước; dừng chuyển box.")


def apply_conversions(client, scope, updates, additions, originals, settings, *, confirmed=False, schema_verified=False):
    if not confirmed or not schema_verified:
        raise ValueError("Chuyển box cần xác nhận và schema hợp lệ trước mọi thao tác ghi.")
    revalidate_scope(client, scope)
    current = client.annotations()
    by_id = {shape["id"]: shape for shape in current.get("shapes", [])}
    if any(by_id.get(key) != value for key, value in originals.items()):
        raise ValueError("Shape được chọn đã bị sửa sau xem trước; không ghi.")
    # Preserve original geometry until every additional component is saved.
    if additions:
        apply_shapes(client, additions, scope, settings, confirmed=True, schema_verified=True)
    changed = 0
    for start in range(0, len(updates), 100):
        revalidate_scope(client, scope)
        current = client.annotations()
        by_id = {shape["id"]: shape for shape in current.get("shapes", [])}
        chunk = updates[start:start + 100]
        if any(by_id.get(shape["id"]) != originals[shape["id"]] for shape in chunk):
            raise ValueError("Shape đã thay đổi trong lúc chuyển box; dừng, không gửi lại.")
        before = annotation_snapshot(current)
        for shape in chunk:
            before["shapes"].pop(shape["id"])
        result = client.update_shape_geometry(chunk, originals=originals, scope=scope,
                                              schema_verified=True, confirmed=True)
        if {s.get("id") for s in result.get("shapes", [])} != {s["id"] for s in chunk}:
            raise ValueError("Server không xác nhận đúng ID cập nhật; dừng, không gửi lại.")
        after = client.annotations()
        verify_preserved(before, after)
        saved = {shape["id"]: shape for shape in after.get("shapes", [])}
        if any(saved.get(shape["id"]) != shape for shape in chunk):
            raise ValueError("Đọc lại chưa xác minh được box đã đổi; dừng, không gửi lại.")
        changed += len(chunk)
        print(f"[OK] Đã đổi và đọc lại xác minh {changed} shape thành box; giữ ID và các box cũ.")
    return changed


def run_conversion(client, scope, job, settings, args):
    if job.get("parent_job_id") or job.get("replicas_count", 0):
        raise ValueError("Chưa hỗ trợ chuyển geometry của Job consensus.")
    scope.allowed_shape_types = {"rectangle"}
    scope.allowed_sources = {"auto", "file"}
    print("[CHUYỂN BOX] Chỉ đổi mask/polygon/polyline có source=auto/file; giữ nguyên box và nhãn vẽ tay.")
    print(f"Mask rời được tách box theo vùng; bỏ nhiễu dưới {settings.semantic_min_pixels} pixel, giữ ít nhất vùng lớn nhất.")
    limits = (1, 5, None) if args.all and args.apply else (args.limit,)
    schema = None
    total = 0
    for limit in limits:
        numbers = sorted(scope.frames)[:limit]
        annotations = client.annotations()
        updates, additions, originals, skipped = plan_conversions(annotations, scope, numbers, settings.semantic_min_pixels)
        print(f"[XEM TRƯỚC] {len(numbers)} frame: đổi {len(updates)} shape; thêm tối đa {len(additions)} box vùng rời; giữ {skipped} shape vẽ tay.")
        for shape in updates:
            print(f"ID {shape['id']} | frame={shape['frame']} | label={client.redactor.clean(scope.labels[shape['label_id']]['name'])} | "
                  f"{originals[shape['id']]['type']} -> rectangle | xyxy={shape['points']}")
        if not args.apply:
            print("[OK] DRY RUN chuyển box hoàn tất; không sửa annotation.")
            continue
        if not updates:
            continue
        if not sys.stdin.isatty():
            raise ValueError("Chuyển box cần Terminal tương tác và xác nhận.")
        if schema is None:
            schema = client.schema()
            verify_create_schema(schema)
            verify_patch_schema(schema, action="update")
        phrase = f"CONVERT JOB {scope.job_id} TO BOXES"
        print("Vùng tô/đường gốc được thay bằng box; tên nhãn, frame, ID gốc và box cũ được giữ.")
        if input(f"Nhập chính xác '{phrase}' để xác nhận: ").strip() != phrase:
            print("Đã hủy chuyển box; dừng các giai đoạn còn lại.")
            return 2
        total += apply_conversions(client, scope, updates, additions, originals, settings,
                                   confirmed=True, schema_verified=True)
    if args.apply:
        print(f"[OK] Chuyển box hoàn tất; đã đổi {total} shape tự động trong Job {scope.job_id}.")
    print(f"Mở CVAT: {client.config.url.rstrip('/')}/tasks/{client.config.task_id}/jobs/{scope.job_id}")
    return 0
