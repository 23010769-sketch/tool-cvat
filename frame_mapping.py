"""Chuyển metadata Job CVAT 2.76.0 sang frame ID của Task, không lấy ảnh ngoài Job."""

import math
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FrameInfo:
    number: int
    width: int
    height: int


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} không hợp lệ.")
    return value


def build_frames(job, metadata):
    start = integer(job.get("start_frame"), "Job start")
    stop = integer(job.get("stop_frame"), "Job stop")
    if stop < start:
        raise ValueError("Khoảng frame Job bị đảo.")
    expression = metadata.get("frame_filter", "").strip()
    match = re.fullmatch(r"step\s*=\s*([1-9]\d*)", expression) if expression else None
    if expression and not match:
        raise ValueError("Không hiểu frame_filter; dừng để tránh lệch frame.")
    step = int(match[1]) if match else 1
    absolute_start = integer(metadata.get("start_frame"), "Metadata start")
    absolute_stop = integer(metadata.get("stop_frame"), "Metadata stop")
    task_origin = absolute_start - start * step
    if task_origin < 0 or not absolute_start <= absolute_stop < task_origin + (stop + 1) * step:
        raise ValueError("Metadata không khớp khoảng frame Job.")
    if absolute_stop < task_origin + stop * step:
        raise ValueError("Metadata stop không đủ frame Job.")

    included = metadata.get("included_frames")
    if included is None:
        numbers = list(range(start, stop + 1))
    else:
        numbers = []
        for absolute in included:
            absolute = integer(absolute, "Included frame")
            relative, remainder = divmod(absolute - task_origin, step)
            if remainder or not start <= relative <= stop:
                raise ValueError("Included frame không khớp step/phạm vi.")
            numbers.append(relative)
        if len(numbers) != len(set(numbers)):
            raise ValueError("Metadata có included frame trùng.")
        numbers.sort()
    if integer(metadata.get("size"), "Metadata size") != len(numbers):
        raise ValueError("Metadata size/included_frames không khớp.")
    deleted = {integer(n, "Deleted frame") for n in metadata.get("deleted_frames", [])}
    media = metadata.get("frames") or []
    if len(media) != stop - start + 1 and not (job.get("mode") == "interpolation" and len(media) == 1):
        raise ValueError("Số phần tử metadata ảnh không khớp; cần kiểm tra schema trước khi tiếp tục.")
    result = {}
    for number in numbers:
        if number in deleted:
            continue
        item = media[0] if len(media) == 1 else media[number - start]
        if item.get("name", "").replace("\\", "/").rsplit("/", 1)[-1] == "placeholder.jpg":
            continue
        width = integer(item.get("width"), "Image width", 1)
        height = integer(item.get("height"), "Image height", 1)
        result[number] = FrameInfo(number, width, height)
    if not result:
        raise ValueError("Job không còn frame hợp lệ để xử lý.")
    return result


@dataclass
class JobScope:
    job_id: int
    frames: dict
    labels: dict
    allowed_shape_types: set = field(default_factory=lambda: {"rectangle"})
    label_geometry: dict = field(default_factory=dict)
    allowed_sources: set = field(default_factory=lambda: {"auto"})

    def validate_shape(self, shape):
        if set(shape) - {"frame", "label_id", "type", "points", "source", "attributes", "occluded", "rotation", "z_order", "score"}:
            raise ValueError("Shape có field không được phép (ID/group/track/element).")
        frame = shape.get("frame")
        label_id = shape.get("label_id")
        if type(frame) is not int or frame not in self.frames:
            raise ValueError("Annotation ngoài các frame hợp lệ của Job.")
        if type(label_id) is not int or label_id not in self.labels:
            raise ValueError("Label ID không thuộc Job.")
        geometry = shape.get("type")
        if geometry not in self.allowed_shape_types or geometry not in ("rectangle", "mask", "polyline"):
            raise ValueError("Geometry không thuộc pipeline đã xác minh.")
        if label_id in self.label_geometry and self.label_geometry[label_id] != geometry:
            raise ValueError("Geometry khác mapping của label đã xác minh.")
        if self.labels[label_id].get("type", "any") not in (geometry, "any"):
            raise ValueError("Label không cho phép geometry được yêu cầu.")
        if shape.get("source") not in self.allowed_sources or shape.get("rotation", 0) != 0:
            raise ValueError("Source không thuộc pipeline hoặc rotation khác 0.")
        attributes = shape.get("attributes", [])
        expected = {a["id"]: str(a["default_value"]) for a in self.labels[label_id].get("attributes", [])}
        actual = {a.get("spec_id"): a.get("value") for a in attributes}
        if len(actual) != len(attributes) or actual != expected:
            raise ValueError("Thuộc tính annotation không khớp default của label trong Job.")
        points = shape.get("points", [])
        info = self.frames[frame]
        if not all(type(p) in (float, int) and math.isfinite(p) for p in points):
            raise ValueError("Tọa độ/RLE phải là số hữu hạn.")
        if geometry == "rectangle":
            if len(points) != 4:
                raise ValueError("Rectangle cần 4 tọa độ.")
            x1, y1, x2, y2 = points
            if not (0 <= x1 < x2 <= info.width and 0 <= y1 < y2 <= info.height):
                raise ValueError("Bounding box ngoài kích thước ảnh hoặc có diện tích bằng 0.")
        elif geometry == "polyline":
            if len(points) < 4 or len(points) % 2 or len(set(zip(points[::2], points[1::2]))) < 2:
                raise ValueError("Polyline cần ít nhất hai điểm khác nhau.")
            if not all(0 <= x <= info.width and 0 <= y <= info.height
                       for x, y in zip(points[::2], points[1::2])):
                raise ValueError("Polyline nằm ngoài ảnh.")
        else:
            if len(points) < 5 or not all(int(p) == p for p in points):
                raise ValueError("Mask RLE không hợp lệ.")
            *runs, x1, y1, x2, y2 = [int(p) for p in points]
            if not (0 <= x1 <= x2 < info.width and 0 <= y1 <= y2 < info.height):
                raise ValueError("Mask bbox nằm ngoài ảnh.")
            if (any(n < 0 for n in runs) or sum(runs) != (x2 - x1 + 1) * (y2 - y1 + 1)
                    or sum(runs[1::2]) <= 0):
                raise ValueError("Mask rỗng hoặc độ dài RLE không khớp bbox.")
        score = shape.get("score", 1.0)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Confidence không hợp lệ.")


def validate_job(config, job, organization=None):
    if job.get("id") != config.job_id or job.get("task_id") != config.task_id or job.get("type") != "annotation":
        raise ValueError("Job ID/Task ID/loại Job khác phạm vi đã chỉ định.")
    if job.get("dimension") != "2d" or job.get("media_type") != "image":
        raise ValueError("Chỉ hỗ trợ Job ảnh/video 2D.")
    if config.organization:
        if not organization or organization.get("slug") != config.organization or organization.get("id") != job.get("organization"):
            raise ValueError("Organization thực của Job khác cấu hình.")
    elif job.get("organization") is not None:
        raise ValueError("Job vẫn thuộc Organization; dùng --organization cohort-5a.")
