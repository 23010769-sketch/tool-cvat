"""YOLO cục bộ, CPU, nhận ảnh trong RAM; không upload hoặc lưu ảnh."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    points: tuple
    geometry: str = "rectangle"
    source: str = "auto"


class YoloDetector:
    def __init__(self, config):
        self.config = config
        # Keep local Ultralytics settings in the project and disable telemetry before import.
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).parent / ".ultralytics"))
        os.environ["YOLO_AUTOINSTALL"] = "false"
        from ultralytics import YOLO, settings
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES, attempt_download_asset
        settings.update({key: False for key in ("sync", "wandb", "mlflow", "clearml", "comet", "dvc", "raytune")
                         if key in settings})
        model_path = Path(config.model).expanduser()
        if not model_path.is_file():
            if model_path.name not in GITHUB_ASSETS_NAMES:
                raise ValueError("Tên weights không có trong danh sách assets của Ultralytics đã cài; dùng file cục bộ hợp lệ.")
            # Download model weights only; never pass CVAT frames/credentials to a download API.
            model_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = Path(attempt_download_asset(str(model_path)))
            if not downloaded.is_file():
                raise ValueError("Không tải được weights; hãy tải từ Ultralytics assets rồi đặt vào models/.")
            model_path = downloaded
        self.model = YOLO(str(model_path))
        if self.model.task != "detect":
            raise ValueError("Weights không phải YOLO Object Detection.")
        self.names = dict(self.model.names)
        self.shape_types = {key: "rectangle" for key in self.names}
        with model_path.open("rb") as stream:
            self.weights_hash = hashlib.file_digest(stream, "sha256").hexdigest()

    def predict(self, images):
        results = self.model.predict(
            source=images, conf=self.config.confidence, iou=self.config.iou,
            device="cpu", imgsz=self.config.image_size, batch=self.config.batch_size,
            save=False, save_txt=False, save_conf=False, save_crop=False,
            show=False, verbose=False, stream=False,
        )
        if len(results) != len(images):
            raise ValueError("YOLO trả số kết quả khác số frame gửi vào.")
        batches = []
        for image, result in zip(images, results):
            if tuple(result.orig_shape) != (image.height, image.width):
                raise ValueError("Kích thước gốc trong kết quả YOLO không khớp ảnh CVAT.")
            boxes = result.boxes
            batches.append([] if boxes is None else [
                Detection(int(cls), float(conf), tuple(float(p) for p in xyxy))
                for cls, conf, xyxy in zip(boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.xyxy.cpu().tolist())
            ])
        return batches
