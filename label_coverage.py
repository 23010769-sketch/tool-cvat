"""Đối chiếu tất cả labels với lớp thật của model và geometry pipeline đang hỗ trợ."""

import json
from pathlib import Path

from label_mapper import LabelMapper, label_candidates, normalize


GEOMETRY_HELP = {
    "polygon": "Cần model segmentation phù hợp và writer polygon; pipeline hiện chỉ tạo rectangle.",
    "mask": "Cần model segmentation phù hợp và writer mask; pipeline hiện chỉ tạo rectangle.",
    "polyline": "Cần model đường/lane phù hợp và writer polyline; pipeline hiện chỉ tạo rectangle.",
    "skeleton": "Cần model pose/keypoints phù hợp và writer skeleton.",
    "points": "Cần model keypoints/points phù hợp và writer points.",
    "tag": "Cần model/logic phân loại và writer tag thay vì rectangle.",
    "cuboid": "Cần model và writer 3D; YOLO detection 2D không tạo cuboid.",
}


def label_coverage(names, labels, overrides=None, *, shape_types=None):
    candidates = label_candidates(names, labels, overrides, shape_types=shape_types)
    base_candidates = label_candidates(names, labels, shape_types=shape_types)
    allowed_types = {"any", "rectangle"} | set((shape_types or {}).values())
    rows = []
    for label in labels:
        label_id, name = label["id"], label["name"]
        shape_type = label.get("type", "any")
        row = {"id": label_id, "name": name, "type": shape_type, "status": "unsupported", "sources": [], "reason": ""}
        if shape_type not in allowed_types or label.get("parent"):
            row["reason"] = GEOMETRY_HELP.get(shape_type, "Geometry/sublabel này chưa được pipeline hỗ trợ.")
        else:
            exact = [names[k] for k, matches in candidates.items() if len(matches) == 1 and matches[0]["id"] == label_id]
            ambiguous = [names[k] for k, matches in candidates.items()
                         if len(matches) > 1 and any(item["id"] == label_id for item in matches)]
            previously_matching = [names[k] for k, matches in base_candidates.items()
                                   if any(item["id"] == label_id for item in matches)]
            if exact:
                row["sources"] = exact
                try:
                    LabelMapper.attributes(label)
                except ValueError as error:
                    row["status"], row["reason"] = "attributes", str(error)
                else:
                    row["status"] = "supported"
                    row["reason"] = "Có lớp model, mapping rõ ràng và writer phù hợp; kết quả vẫn phụ thuộc ảnh/confidence."
            elif ambiguous:
                row["status"], row["sources"] = "ambiguous", ambiguous
                row["reason"] = "Cần chọn đúng label ID; không nhân một dự đoán sang nhiều nhãn khác nghĩa."
            elif previously_matching:
                row["status"] = "not_selected"
                row["reason"] = "Mapping hiện tại đã bỏ qua lớp hoặc chọn label đích khác."
            else:
                name_key = normalize(name)
                if any(word in name_key for word in ("lane", "drivable", "segmentation")):
                    row["reason"] = "Model object detection hiện tại không có lớp này; cần model phù hợp riêng."
                else:
                    row["reason"] = "Chưa có lớp model/mapping tương ứng; cần mapping đúng nghĩa hoặc weights hỗ trợ lớp này."
        rows.append(row)
        for child in label.get("sublabels", []):
            rows.append({"id": child["id"], "name": f"{name}/{child['name']}", "type": child.get("type", "points"),
                         "status": "unsupported", "sources": [], "reason": "Sublabel phải xử lý cùng geometry/model của label cha."})
    return rows


def print_label_coverage(rows, redactor):
    print("=== LABEL COVERAGE ===")
    for row in rows:
        print(f"ID {row['id']} | {redactor.clean(row['name'])} | type={redactor.clean(row['type'])} | {row['status']}")
        if row["sources"]:
            print("  YOLO:", redactor.clean(", ".join(row["sources"])))
        print(" ", redactor.clean(row["reason"]))
    supported = sum(row["status"] == "supported" for row in rows)
    print(f"Hỗ trợ: {supported}/{len(rows)} labels; còn {len(rows) - supported} label cần xử lý.")
    print("Báo cáo hỗ trợ không bảo đảm model phát hiện đủ mọi đối tượng trên ảnh.")


def require_full_coverage(rows):
    missing = [row for row in rows if row["status"] != "supported"]
    if missing:
        ids = ", ".join(str(row["id"]) for row in missing)
        raise ValueError(f"Chưa hỗ trợ tất cả labels (ID: {ids}). Dừng trước khi đọc ảnh/ghi annotation; xem LABEL COVERAGE.")


def save_label_audit(config, names, rows, redactor, *, directory=None):
    """Save label metadata only, so a later code session can inspect IDs/types."""
    directory = Path(directory) if directory is not None else Path(__file__).parent / "outputs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"job_{config.job_id}_label_audit.json"
    clean_rows = [{"id": row["id"], "name": redactor.clean(row["name"]), "type": redactor.clean(row["type"]),
                   "status": row["status"], "sources": [redactor.clean(s) for s in row["sources"]],
                   "reason": redactor.clean(row["reason"])} for row in rows]
    report = {"server": config.url, "organization": config.organization, "task_id": config.task_id,
              "job_id": config.job_id, "model_classes": {str(k): redactor.clean(v) for k, v in names.items()},
              "labels": clean_rows, "contains_images": False, "contains_credentials": False}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path
