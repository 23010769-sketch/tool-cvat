"""Chuẩn bị rectangle/mask/polyline, chỉ PATCH/create sau xác nhận."""

import math


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def existing_boxes(annotations, frame):
    for shape in annotations.get("shapes", []):
        if shape.get("frame") == frame and shape.get("type") == "rectangle" and not shape.get("outside", False):
            # Existing axis-aligned rectangles use pixel IoU below.
            points = shape.get("points", [])
            if len(points) == 4:
                yield shape.get("label_id"), points
    for track in annotations.get("tracks", []):
        shapes = sorted(track.get("shapes", []), key=lambda s: s["frame"])
        left = next((s for s in reversed(shapes) if s["frame"] <= frame), None)
        if not left or left.get("outside") or left.get("type") != "rectangle":
            continue
        points = left.get("points", [])
        if len(points) != 4:
            continue
        right = next((s for s in shapes if s["frame"] > frame), None)
        if right and right.get("type") == "rectangle" and len(right.get("points", [])) == 4:
            factor = (frame - left["frame"]) / (right["frame"] - left["frame"])
            points = [a + (b - a) * factor for a, b in zip(points, right["points"])]
        yield track.get("label_id"), points


def _bitmap(shape, info):
    import cv2
    import numpy as np
    from cvat_sdk.masks import decode_mask
    geometry, points = shape.get("type"), shape.get("points", [])
    if geometry == "mask":
        return decode_mask(points, image_width=info.width, image_height=info.height)
    image = np.zeros((info.height, info.width), dtype=np.uint8)
    if geometry == "polygon" and len(points) >= 6 and len(points) % 2 == 0:
        vertices = np.rint(np.asarray(points).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(image, [vertices], 1)
    elif geometry == "rectangle" and len(points) == 4 and shape.get("rotation", 0) == 0:
        x1, y1, x2, y2 = points
        image[max(0, math.floor(y1)):min(info.height, math.ceil(y2)),
              max(0, math.floor(x1)):min(info.width, math.ceil(x2))] = 1
    else:
        raise ValueError("Chưa hỗ trợ bảo toàn semantic geometry/rotation này; dừng trước khi ghi mask.")
    return image.astype(bool)


def deduplicate(candidates, annotations, threshold, *, scope=None):
    index = {}
    reserved = {}
    accepted, duplicates = [], 0
    lines = {}
    for shape in candidates:
        frame = shape["frame"]
        if shape.get("type") == "polyline":
            if scope is None:
                raise ValueError("Polyline cần phạm vi đã xác minh.")
            scope.validate_shape(shape)
            key = (frame, shape["label_id"])
            if key not in lines:
                if any(t.get("label_id") == shape["label_id"] for t in annotations.get("tracks", [])):
                    raise ValueError("Chưa hỗ trợ chống trùng lane track; dừng trước khi ghi.")
                old = [s for s in annotations.get("shapes", []) if s.get("frame") == frame
                       and s.get("label_id") == shape["label_id"] and not s.get("outside", False)]
                if any(s.get("type") != "polyline" or s.get("rotation", 0) != 0 for s in old):
                    raise ValueError("Lane cũ có geometry khác polyline; cần kiểm tra trước khi thêm.")
                lines[key] = [s["points"] for s in old]
            if any(same_polyline(shape["points"], points) for points in lines[key]):
                duplicates += 1
                continue
            accepted.append(shape)
            lines[key].append(shape["points"])
            continue
        if shape.get("type") == "mask":
            import numpy as np
            from cvat_sdk.masks import encode_mask
            if scope is None:
                raise ValueError("Chống trùng mask cần kích thước frame đã xác minh.")
            scope.validate_shape(shape)
            info = scope.frames[frame]
            if frame not in reserved:
                protected_ids = {key for key, value in scope.label_geometry.items() if value == "mask"} or {shape["label_id"]}
                if any(track.get("label_id") in protected_ids for track in annotations.get("tracks", [])):
                    raise ValueError("Chưa hỗ trợ bảo toàn semantic track khi tạo mask mới; dừng.")
                reserved[frame] = np.zeros((info.height, info.width), dtype=bool)
                for old in annotations.get("shapes", []):
                    if old.get("frame") == frame and old.get("label_id") in protected_ids and not old.get("outside", False):
                        reserved[frame] |= _bitmap(old, info)
            bitmap = _bitmap(shape, info)
            remaining = bitmap & ~reserved[frame]
            if not remaining.any():
                duplicates += 1
                continue
            # Add only pixels not covered by any existing semantic annotation.
            accepted.append({**shape, "points": encode_mask(remaining)})
            reserved[frame] |= remaining
            continue
        if frame not in index:
            index[frame] = list(existing_boxes(annotations, frame))
        # Do not add same-class boxes to frames with a rotated rectangle of that
        # class: polygon overlap is intentionally not approximated as axis IoU.
        rotated = any(s.get("frame") == frame and s.get("label_id") == shape["label_id"]
                      and s.get("type") == "rectangle" and s.get("rotation", 0) != 0
                      for s in annotations.get("shapes", []))
        rotated = rotated or any(t.get("label_id") == shape["label_id"]
                                  and any(s.get("frame", frame + 1) <= frame and s.get("rotation", 0) != 0
                                          for s in t.get("shapes", []))
                                  for t in annotations.get("tracks", []))
        if rotated:
            duplicates += 1
            continue
        if any(label_id == shape["label_id"] and iou(points, shape["points"]) >= threshold
               for label_id, points in index[frame]):
            duplicates += 1
            continue
        accepted.append(shape)
        index[frame].append((shape["label_id"], shape["points"]))
    return accepted, duplicates


def same_polyline(a, b, tolerance=0.5):
    """Geometric comparison tolerates reversed direction and different sampling."""
    import numpy as np

    def vertices(points):
        array = np.asarray(points, dtype=float)
        if array.size < 4 or array.size % 2 or not np.isfinite(array).all():
            raise ValueError("Polyline cũ không hợp lệ; dừng chống trùng.")
        array = array.reshape(-1, 2)
        if len(array) > 4096:
            raise ValueError("Polyline vượt giới hạn kiểm tra chống trùng.")
        return array

    a, b = vertices(a), vertices(b)
    if np.any(np.abs(a.min(axis=0) - b.min(axis=0)) > tolerance) or np.any(np.abs(a.max(axis=0) - b.max(axis=0)) > tolerance):
        return False

    def covered(source, target):
        delta = target[1:] - target[:-1]
        denom = (delta * delta).sum(axis=1)
        for left, right in zip(source[:-1], source[1:]):
            count = max(2, int(np.ceil(np.linalg.norm(right - left) / tolerance)) + 1)
            if count > 20000:
                raise ValueError("Polyline vượt giới hạn lấy mẫu.")
            samples = left + np.linspace(0., 1., count)[:, None] * (right - left)
            for start in range(0, count, 128):
                block = samples[start:start + 128]
                offsets = block[:, None, :] - target[:-1]
                ratio = np.divide((offsets * delta).sum(axis=2), denom,
                                  out=np.zeros(offsets.shape[:2]), where=denom != 0)
                closest = target[:-1] + np.clip(ratio, 0., 1.)[:, :, None] * delta
                distances = ((block[:, None, :] - closest) ** 2).sum(axis=2).min(axis=1)
                if (distances > tolerance ** 2).any():
                    return False
        return True

    return covered(a, b) and covered(b, a)


def make_shape(frame, detection, label, mapper, scope):
    points = list(detection.points)
    geometry = getattr(detection, "geometry", "rectangle")
    if geometry == "rectangle":
        if len(points) != 4 or not all(math.isfinite(p) for p in points):
            raise ValueError("YOLO trả tọa độ không hợp lệ.")
        info = scope.frames[frame]
        points = [max(0., min(float(points[0]), info.width)), max(0., min(float(points[1]), info.height)),
                  max(0., min(float(points[2]), info.width)), max(0., min(float(points[3]), info.height))]
    shape = {"frame": frame, "label_id": label["id"], "type": geometry, "points": points,
             "source": getattr(detection, "source", "auto"), "rotation": 0., "occluded": False, "z_order": 0,
             "score": detection.confidence, "attributes": mapper.attributes(label)}
    if shape["source"] == "file":
        shape.pop("score")  # Reference labels do not have a model confidence.
    scope.validate_shape(shape)
    return shape


def verify_create_schema(schema, *, required_types=("rectangle",), required_sources=None):
    return verify_patch_schema(schema, action="create", required_types=required_types, required_sources=required_sources)


def verify_patch_schema(schema, *, action, required_types=("rectangle",), required_sources=None):
    if action not in ("create", "update"):
        raise ValueError("Chỉ xác minh create/update chính thức.")
    if schema.get("info", {}).get("version") not in ("2.76.0", "2.76.0 (2.0)"):
        raise ValueError("Schema server khác 2.76.0; chặn ghi.")

    def resolve(value):
        seen = set()
        while "$ref" in value:
            ref = value["$ref"]
            if not ref.startswith("#/") or ref in seen:
                raise ValueError("Schema reference không hỗ trợ; chặn ghi.")
            seen.add(ref)
            value = schema
            for token in ref[2:].split("/"):
                value = value[token.replace("~1", "/").replace("~0", "~")]
        return value

    paths = schema.get("paths", {})
    operation = next((paths[p].get("patch") for p in ("/api/jobs/{id}/annotations/", "/api/jobs/{id}/annotations")
                      if p in paths), None)
    if not operation or operation.get("operationId") != "jobs_partial_update_annotations":
        raise ValueError("Không xác minh được PATCH annotation chính thức của Job.")
    action_parameter = next((resolve(p) for p in operation.get("parameters", [])
                   if resolve(p).get("name") == "action" and resolve(p).get("in") == "query"), None)
    if not action_parameter or not action_parameter.get("required") or action not in resolve(action_parameter.get("schema", {})).get("enum", []):
        raise ValueError(f"Schema không xác nhận action={action}; chặn ghi.")
    body = resolve(operation.get("requestBody", {}))
    payload = resolve(body.get("content", {}).get("application/json", {}).get("schema", {}))
    shapes = resolve(payload.get("properties", {}).get("shapes", {}))
    rectangle = resolve(shapes.get("items", {}))
    properties = rectangle.get("properties", {})
    if shapes.get("type") != "array" or not {"frame", "label_id", "type", "points", "source"} <= set(properties):
        raise ValueError("Schema shapes khác dự kiến; chặn ghi.")
    if action == "update" and "id" not in properties:
        raise ValueError("Schema update không có ID annotation; chặn ghi.")
    if not set(required_types) <= set(resolve(properties["type"]).get("enum", [])):
        raise ValueError("Schema không hỗ trợ geometry pipeline yêu cầu; chặn ghi.")
    if required_sources is not None:
        source = resolve(properties["source"])
        # The real 2.76.0 server exposes source as an unrestricted string, not
        # an enum. Official SourceType defines auto/file; never invent sources.
        if (not set(required_sources) <= {"auto", "file"}
                or source.get("type") != "string"
                or "enum" in source and not set(required_sources) <= set(source["enum"])):
            raise ValueError("Schema không hỗ trợ source của pipeline; chặn ghi.")
    return True


def annotation_snapshot(annotations):
    # Preserve full old objects, not only their IDs. No annotations are written to disk.
    return {kind: {item["id"]: item for item in annotations.get(kind, [])}
            for kind in ("tags", "shapes", "tracks", "intervals")}


def verify_preserved(before, after):
    snapshot = annotation_snapshot(after)
    for kind, items in before.items():
        if any(snapshot[kind].get(item_id) != item for item_id, item in items.items()):
            raise ValueError("Annotation cũ thay đổi trong lúc ghi/kiểm tra. Dừng; không tự rollback hoặc gửi lại.")


def apply_shapes(client, candidates, scope, config, *, confirmed, schema_verified):
    if not confirmed or not schema_verified:
        raise ValueError("Cần xác nhận và schema hợp lệ trước PATCH.")
    # Re-read Job and metadata to catch changes while the detector was running.
    from frame_mapping import build_frames, validate_job
    job = client.job()
    organization = client.organization(job["organization"]) if client.config.organization else None
    validate_job(client.config, job, organization)
    if job.get("parent_job_id") or job.get("replicas_count", 0):
        raise ValueError("Chưa hỗ trợ ghi Job có consensus replicas.")
    if build_frames(job, client.metadata()) != scope.frames:
        raise ValueError("Metadata frame thay đổi sau khi xem trước; chạy lại dry-run.")
    current_labels = {label["id"]: label for label in client.job_labels()}
    if current_labels != scope.labels:
        raise ValueError("Labels thay đổi sau khi xem trước; chạy lại dry-run.")
    current = client.annotations()
    pending, duplicate_count = deduplicate(candidates, current, config.duplicate_iou, scope=scope)
    created = 0
    for start in range(0, len(pending), 100):
        # Re-read per request and compare snapshots to preserve annotations from other editors.
        current = client.annotations()
        chunk, duplicates = deduplicate(pending[start:start + 100], current, config.duplicate_iou, scope=scope)
        duplicate_count += duplicates
        if not chunk:
            continue
        before = annotation_snapshot(current)
        result = client.create_shapes(chunk, scope=scope, schema_verified=True, confirmed=True)
        new_shapes = result.get("shapes", [])
        if len(new_shapes) != len(chunk) or not all(type(s.get("id")) is int for s in new_shapes):
            raise ValueError("Server không xác nhận đủ shape vừa tạo; dừng, không gửi lại.")
        after = client.annotations()
        verify_preserved(before, after)
        by_id = {shape["id"]: shape for shape in after.get("shapes", [])}
        for saved, expected in zip(new_shapes, chunk):
            actual = by_id.get(saved["id"], {})
            if (actual.get("frame") != expected["frame"] or actual.get("label_id") != expected["label_id"]
                    or actual.get("type") != expected["type"] or actual.get("points") != expected["points"]):
                raise ValueError("Đọc lại không xác minh được annotation vừa tạo; dừng, không gửi lại.")
        created += len(new_shapes)
        print(f"[OK] Đã xác minh thêm {created} shape; annotation cũ giữ nguyên tại lần đọc lại.")
    return created, duplicate_count
