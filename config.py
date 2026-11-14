"""Cấu hình chẩn đoán; không đọc hoặc ghi bí mật vào file."""

import argparse
import os
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Config:
    url: str = "https://cvat.edufun.ai"
    organization: str = "cohort-5a"
    task_id: int = 71
    job_id: int = 206
    timeout: float = 30
    frame_timeout: float = 120
    backend: str = "sdk"
    auth: str = "auto"

    def __post_init__(self):
        parsed = urlsplit(self.url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ValueError("CVAT_URL phải là HTTPS origin, không chứa /api, tài khoản hoặc query.")
        if self.task_id <= 0 or self.job_id <= 0 or not 0 < self.timeout <= 300:
            raise ValueError("Task/Job phải dương; timeout trong khoảng (0, 300].")
        if not math.isfinite(self.frame_timeout) or not 0 < self.frame_timeout <= 600:
            raise ValueError("frame_timeout phải trong khoảng (0, 600].")
        object.__setattr__(self, "organization", self.organization.strip())
        if self.backend not in ("sdk", "rest") or self.auth not in ("auto", "password", "pat", "session"):
            raise ValueError("Backend hoặc phương thức xác thực không hợp lệ.")


@dataclass(frozen=True)
class DetectionConfig:
    model: str = "models/yolo11n.pt"
    confidence: float = 0.45
    iou: float = 0.50
    duplicate_iou: float = 0.85
    batch_size: int = 1
    image_size: int = 640
    device: str = "cpu"
    pipeline: str = "detection"
    semantic_model: str = "models/segformer-b0-cityscapes"
    semantic_image_size: int = 640
    semantic_confidence: float = 0.50
    semantic_min_pixels: int = 64
    output_geometry: str = "native"

    def __post_init__(self):
        for value in (self.confidence, self.iou, self.duplicate_iou):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Confidence/IoU phải trong (0, 1].")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model phải là đường dẫn weights không trống.")
        if self.device != "cpu" or type(self.batch_size) is not int or not 1 <= self.batch_size <= 8:
            raise ValueError("Dùng CPU, batch_size từ 1 đến 8.")
        if type(self.image_size) is not int or not 32 <= self.image_size <= 1920 or self.image_size % 32:
            raise ValueError("image_size phải là bội số 32, từ 32 đến 1920.")
        if self.pipeline not in ("detection", "scene", "bdd-reference", "scene-reference"):
            raise ValueError("pipeline phải là detection, scene, bdd-reference hoặc scene-reference.")
        if self.output_geometry not in ("native", "rectangle"):
            raise ValueError("output_geometry phải là native hoặc rectangle.")
        if self.pipeline == "scene-reference" and self.output_geometry != "rectangle":
            raise ValueError("scene-reference yêu cầu output_geometry=rectangle.")
        if not isinstance(self.semantic_model, str) or not self.semantic_model.strip():
            raise ValueError("semantic_model phải là đường dẫn model không trống.")
        if type(self.semantic_image_size) is not int or not 128 <= self.semantic_image_size <= 1024:
            raise ValueError("semantic_image_size phải từ 128 đến 1024.")
        if (type(self.semantic_confidence) not in (int, float) or not math.isfinite(self.semantic_confidence)
                or not 0 < self.semantic_confidence <= 1 or type(self.semantic_min_pixels) is not int or self.semantic_min_pixels < 1):
            raise ValueError("Ngưỡng semantic/confidence hoặc min_pixels không hợp lệ.")

    def checkpoint_dict(self):
        values = asdict(self)
        if self.pipeline == "detection":
            # Keep checkpoints made by the original detection-only program valid.
            return {k: values[k] for k in ("model", "confidence", "iou", "duplicate_iou", "batch_size", "image_size", "device")}
        return values


def load_detection_config(path):
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    if set(data) - {"yolo", "labels"}:
        raise ValueError("Config chỉ hỗ trợ mục [yolo] và [labels].")
    aliases = data.get("labels", {})
    if not all(isinstance(k, str) and (isinstance(v, str) or type(v) is int and v > 0)
               for k, v in aliases.items()):
        raise ValueError("[labels] phải ánh xạ tên YOLO sang tên label CVAT hoặc label ID dương.")
    return DetectionConfig(**data.get("yolo", {})), aliases


def parse_config(argv=None, *, annotation=False):
    parser = argparse.ArgumentParser(description=("YOLO qua Job API; mặc định dry-run, ghi cần --apply và xác nhận."
                                                 if annotation else "Chẩn đoán CVAT từng bước, không sửa annotation."))
    parser.add_argument("--url", default=os.getenv("CVAT_URL", "https://cvat.edufun.ai"))
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--organization", default=os.getenv("CVAT_ORGANIZATION", "cohort-5a"))
    scope.add_argument("--personal", action="store_true", help="Personal Workspace; bỏ Organization header và biến môi trường org.")
    parser.add_argument("--task-id", type=int, default=os.getenv("CVAT_TASK_ID", "71"))
    parser.add_argument("--job-id", type=int, default=os.getenv("CVAT_JOB_ID", "206"))
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--frame-timeout", type=float, default=120,
                        help="Thời gian chờ đọc ảnh, giây; mặc định 120. Không áp dụng cho ghi annotation.")
    parser.add_argument("--backend", choices=("sdk", "rest"), default="sdk")
    parser.add_argument("--auth", choices=("auto", "password", "pat", "session"), default="auto")
    parser.add_argument("--username", default=None)
    parser.add_argument("--public-only", action="store_true", help="Chỉ kiểm tra phiên bản, không đăng nhập.")
    parser.add_argument("--diagnose-permissions", action="store_true",
                        help="Kiểm tra độc lập Organization, membership, Task và Job; chỉ đọc, không tải ảnh.")
    parser.add_argument("--inspect-annotations", action="store_true",
                        help="Đếm annotation thật theo từng label trong Job; không chạy AI, không tải ảnh hoặc ghi dữ liệu.")
    if annotation:
        frames = parser.add_mutually_exclusive_group()
        frames.add_argument("--limit", type=int, choices=(1, 5), help="Thử 1 rồi 5 frame; bỏ tham số để chạy toàn Job.")
        frames.add_argument("--all", action="store_true",
                            help="Hoàn tất các bước còn thiếu rồi xử lý toàn Job trong một lần đăng nhập; ghi cần --apply.")
        parser.add_argument("--apply", action="store_true", help="Thêm annotation sau khi xem kết quả và xác nhận.")
        parser.add_argument("--convert-existing-to-boxes", action="store_true",
                            help="Chuyển mask/polygon/polyline tự động hiện có thành box; giữ box cũ, không chạy model hoặc tải ảnh.")
        parser.add_argument("--config", default=str(Path(__file__).with_name("annotation.toml")))
        parser.add_argument("--audit-labels", action="store_true",
                            help="Đọc labels Job và báo cáo model hỗ trợ lớp nào; không đọc frame hoặc ghi annotation.")
        parser.add_argument("--require-all-labels", action="store_true",
                            help="Dừng trước khi tải frame nếu có label chưa được model/mapping/writer hỗ trợ.")
    args = parser.parse_args(argv)
    if annotation and args.audit_labels and (args.apply or args.all or args.limit is not None or args.convert_existing_to_boxes):
        parser.error("--audit-labels chạy riêng, không kết hợp --apply/--all/--limit.")
    if annotation and args.inspect_annotations and (args.apply or args.all or args.limit is not None or args.audit_labels or args.convert_existing_to_boxes):
        parser.error("--inspect-annotations chạy riêng, không kết hợp chế độ xử lý ảnh hoặc ghi.")
    if args.personal:
        args.organization = ""
    try:
        config = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})
    except ValueError as error:
        parser.error(str(error))
    return config, args
