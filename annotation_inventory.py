"""Kiểm kê annotation đã có theo label, chỉ đọc metadata, không lưu tọa độ hoặc ảnh."""

import json
from collections import Counter
from pathlib import Path

from frame_mapping import validate_job


def summarize_annotations(labels, annotations):
    rows = {}

    def add_labels(items):
        for label in items:
            rows[label["id"]] = {"id": label["id"], "name": label["name"], "type": label.get("type", "any"),
                                  "shapes": 0, "tracks": 0, "track_keyframes": 0, "tags": 0, "intervals": 0,
                                  "shape_types": Counter()}
            add_labels(label.get("sublabels", []))

    add_labels(labels)

    def count_object(item, kind):
        label_id = item["label_id"]
        if label_id not in rows:
            rows[label_id] = {"id": label_id, "name": "[Label ID không có trong danh sách API]", "type": "unknown",
                              "shapes": 0, "tracks": 0, "track_keyframes": 0, "tags": 0, "intervals": 0,
                              "shape_types": Counter()}
        row = rows[label_id]
        row[kind] += 1
        if kind == "shapes":
            row["shape_types"][item.get("type", "unknown")] += 1
        elif kind == "tracks":
            row["track_keyframes"] += len(item.get("shapes", []))
        for element in item.get("elements", []):
            count_object(element, kind)

    for kind in ("shapes", "tracks", "tags", "intervals"):
        for item in annotations.get(kind, []):
            count_object(item, kind)
    for row in rows.values():
        row["shape_types"] = dict(row["shape_types"])
        row["has_annotations"] = any(row[kind] for kind in ("shapes", "tracks", "tags", "intervals"))
    return list(rows.values())


def inspect_annotations(client, config, redactor, *, directory=None):
    job = client.job()
    organization = client.organization(job["organization"]) if config.organization else None
    validate_job(config, job, organization)
    labels = client.job_labels()
    annotations = client.annotations()
    rows = summarize_annotations(labels, annotations)
    cleaned = []
    print("=== ANNOTATION INVENTORY — TOÀN JOB ===")
    for row in rows:
        safe_row = {**row, "name": redactor.clean(row["name"]), "type": redactor.clean(row["type"]),
                    "shape_types": {redactor.clean(key): count for key, count in row["shape_types"].items()}}
        cleaned.append(safe_row)
        status = "ĐÃ CÓ ANNOTATION" if row["has_annotations"] else "CHƯA CÓ ANNOTATION"
        print(f"ID {row['id']} | {safe_row['name']} | type={safe_row['type']} | {status}")
        print(f"  shapes={row['shapes']}; tracks={row['tracks']}; keyframes={row['track_keyframes']}; "
              f"tags={row['tags']}; geometry={safe_row['shape_types']}")
    populated = sum(row["has_annotations"] for row in rows)
    print(f"Labels có annotation: {populated}/{len(rows)}. Đây là số đếm toàn Job, không chỉ ảnh đang mở.")
    print("Danh sách label và trạng thái ẩn/khóa trên UI không chứng minh đã có đối tượng; xem số đếm ở trên.")
    folder = Path(directory) if directory is not None else Path(__file__).parent / "outputs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"job_{config.job_id}_annotation_inventory.json"
    report = {"report_type": "annotation_inventory", "server": config.url, "organization": config.organization,
              "task_id": config.task_id, "job_id": config.job_id, "labels": cleaned,
              "contains_coordinates": False, "contains_images": False, "contains_credentials": False}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(f"Báo cáo: {path}")
    print("[OK] Chỉ kiểm kê; không tải ảnh, chạy AI hoặc sửa annotation.")
    return path
