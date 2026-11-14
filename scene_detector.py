"""BDD detection + Cityscapes semantic masks, hoàn toàn trong RAM/CPU."""

import hashlib
import json
import os
from pathlib import Path

from prepare_bdd_model import file_hash
from prepare_scene_model import FILES
from yolo_detector import Detection, YoloDetector


SEMANTIC_NAMES = ("road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                  "traffic sign", "vegetation", "terrain", "sky", "person")
CITYSCAPES_NAMES = SEMANTIC_NAMES + ("rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")


class SceneDetector:
    def __init__(self, config):
        self.config = config
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ.setdefault("HF_HOME", str(Path(__file__).parent / ".huggingface"))
        self.detector = YoloDetector(config)
        model_path = Path(config.semantic_model)
        for name, (_, expected) in FILES.items():
            path = model_path / name
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError("Thiếu/sai checksum SegFormer; chạy python prepare_scene_model.py trước.")
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError:
            raise ValueError("Thiếu backend semantic; cài requirements-scene.txt trong .venv.") from None
        self.torch = torch
        preprocessor = json.loads((model_path / "preprocessor_config.json").read_text(encoding="utf-8"))
        preprocessor.pop("feature_extractor_type", None)
        if "reduce_labels" in preprocessor:
            preprocessor["do_reduce_labels"] = preprocessor.pop("reduce_labels")
        self.processor = SegformerImageProcessor(**preprocessor)
        self.semantic = SegformerForSemanticSegmentation.from_pretrained(
            str(model_path), local_files_only=True, use_safetensors=False, weights_only=True,
        ).to("cpu").eval()
        model_names = {int(k): v for k, v in self.semantic.config.id2label.items()}
        if [model_names[k] for k in sorted(model_names)] != list(CITYSCAPES_NAMES):
            raise ValueError("Semantic model có class order khác Cityscapes đã kiểm tra.")
        self.names = {key: "det/" + value for key, value in self.detector.names.items()}
        self.shape_types = {key: "rectangle" for key in self.detector.names}
        self.semantic_ids = {key: name for key, name in model_names.items() if name in SEMANTIC_NAMES}
        self.names.update({100 + key: "semantic/" + name for key, name in self.semantic_ids.items()})
        self.shape_types.update({100 + key: "rectangle" if config.output_geometry == "rectangle" else "mask"
                                 for key in self.semantic_ids})
        digest = self.detector.weights_hash + "".join(file_hash(model_path / name) for name in sorted(FILES))
        self.weights_hash = hashlib.sha256(digest.encode()).hexdigest()

    def semantic_predictions(self, image):
        from cvat_sdk.masks import encode_mask
        # Keep aspect ratio approximately; input dimensions are multiples of 32.
        scale = self.config.semantic_image_size / max(image.width, image.height)
        width = max(32, round(image.width * scale / 32) * 32)
        height = max(32, round(image.height * scale / 32) * 32)
        inputs = self.processor(images=image, return_tensors="pt", size={"height": height, "width": width})
        with self.torch.inference_mode():
            logits = self.semantic(**inputs).logits
            logits = self.torch.nn.functional.interpolate(logits, size=(image.height, image.width),
                                                         mode="bilinear", align_corners=False)[0]
            confidence, category = logits.softmax(dim=0).max(dim=0)
        category, confidence = category.cpu().numpy(), confidence.cpu().numpy()
        result = []
        for class_id in self.semantic_ids:
            bitmap = (category == class_id) & (confidence >= self.config.semantic_confidence)
            if int(bitmap.sum()) < self.config.semantic_min_pixels:
                continue
            if self.config.output_geometry == "rectangle":
                import cv2
                import numpy as np
                count, components, stats, _ = cv2.connectedComponentsWithStats(bitmap.astype(np.uint8), connectivity=8)
                for component in range(1, count):
                    x, y, w, h, area = (int(v) for v in stats[component])
                    if area < self.config.semantic_min_pixels:
                        continue
                    score = float(confidence[components == component].mean())
                    result.append(Detection(100 + class_id, score, (float(x), float(y), float(x + w), float(y + h))))
            else:
                score = float(confidence[bitmap].mean())
                # Native output retains disconnected areas and holes in the RLE.
                result.append(Detection(100 + class_id, score, tuple(encode_mask(bitmap)), "mask"))
        return result

    def predict(self, images):
        results = self.detector.predict(images)
        for image, predictions in zip(images, results):
            predictions.extend(self.semantic_predictions(image))
        return results
