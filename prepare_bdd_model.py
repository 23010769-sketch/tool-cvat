"""Tải model detection BDD100K đã ghim revision/checksum; không kết nối CVAT."""

import hashlib
import importlib
from pathlib import Path

import requests
from tqdm import tqdm


REPOSITORY = "dronefreak/bdd100k-yolo11n"
REVISION = "2b5554133522f96f50ce87a4df0a2de556e4b7ad"
WEIGHTS_SHA256 = "c120d075bcc6aaf94523f9a2c9d681819ebd692f36eb5ea64656d1c5f1278a70"
WEIGHTS_SIZE = 5490522
WEIGHTS_URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/best.pt"
DESTINATION = Path(__file__).parent / "models" / "bdd100k-yolo11n.pt"
EXPECTED_CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign"]


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_checked(url, destination, expected_hash, expected_size, *, session):
    destination = Path(destination)
    if destination.exists():
        if file_hash(destination) == expected_hash:
            return destination
        raise ValueError("File model đã tồn tại nhưng checksum khác; không ghi đè file này.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=(15, 120), verify=True) as response:
            response.raise_for_status()
            digest, size = hashlib.sha256(), 0
            with temporary.open("wb") as stream, tqdm(total=expected_size, unit="B", unit_scale=True, desc="BDD weights") as progress:
                for chunk in response.iter_content(256 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > expected_size:
                        raise ValueError("Model tải về lớn hơn kích thước đã xác minh.")
                    digest.update(chunk)
                    stream.write(chunk)
                    progress.update(len(chunk))
            if size != expected_size or digest.hexdigest() != expected_hash:
                raise ValueError("Model tải về sai kích thước/checksum; không sử dụng weights này.")
        if destination.exists():
            if file_hash(destination) != expected_hash:
                raise ValueError("File đích thay đổi trong lúc tải; dừng để không ghi đè.")
            temporary.unlink(missing_ok=True)
        else:
            temporary.replace(destination)
        return destination
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def validate_checkpoint(path):
    """Only allow installed torch/Ultralytics neural-network types in this pickle."""
    import torch
    allowed = []
    for symbol in torch.serialization.get_unsafe_globals_in_checkpoint(path):
        module_name, _, class_name = symbol.rpartition(".")
        if not module_name.startswith(("ultralytics.nn.", "torch.nn.")):
            raise ValueError("Checkpoint có global ngoài nhóm neural-network được phép; không nạp.")
        value = getattr(importlib.import_module(module_name), class_name)
        if not isinstance(value, type) or not issubclass(value, torch.nn.Module):
            raise ValueError("Checkpoint có callable không phải neural-network class; không nạp.")
        allowed.append(value)
    with torch.serialization.safe_globals(allowed):
        torch.load(path, map_location="cpu", weights_only=True)


def main():
    from config import DetectionConfig
    from yolo_detector import YoloDetector
    import os
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).parent / ".ultralytics"))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    try:
        with requests.Session() as session:
            path = download_checked(WEIGHTS_URL, DESTINATION, WEIGHTS_SHA256, WEIGHTS_SIZE, session=session)
        validate_checkpoint(path)
        detector = YoloDetector(DetectionConfig(model=str(path)))
        classes = [detector.names[key] for key in sorted(detector.names)]
        if classes != EXPECTED_CLASSES:
            raise ValueError("Danh sách lớp thực tế khác model card đã kiểm tra; dừng.")
        from PIL import Image
        with Image.new("RGB", (640, 480)) as image:
            detector.predict([image])
        print("[OK] Model BDD100K nano: checksum, 10 lớp và CPU inference ảnh trống đều thành công.")
        print("Classes:", ", ".join(classes))
        print("Model này chỉ object detection, không xử lý drivable/lane/semantic segmentation.")
        print("Cấu hình: bdd_detection.toml. Chưa kết nối CVAT hoặc ghi annotation.")
        return 0
    except Exception as error:
        print(f"[DỪNG] {type(error).__name__}: không chuẩn bị được weights; file đích chưa được xác minh thì không sử dụng.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
