"""Mask RLE, phân biệt labels semantic/detection và bảo toàn annotation bằng dữ liệu giả."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from cvat_sdk.masks import decode_mask, encode_mask

from annotation_writer import apply_shapes, deduplicate, make_shape, verify_create_schema
from auth import Redactor
from config import Config, DetectionConfig, load_detection_config
from cvat_client import CvatClient
from frame_mapping import FrameInfo, JobScope
from label_coverage import label_coverage
from label_mapper import LabelMapper
from scene_detector import SceneDetector, SEMANTIC_NAMES
from yolo_detector import Detection
from main import main
from run_state import RunState
from test_annotation import valid_schema


LABELS = {637: {"id": 637, "name": "pedestrian", "type": "any", "attributes": []},
          1750: {"id": 1750, "name": "road", "type": "any", "attributes": []},
          1751: {"id": 1751, "name": "sidewalk", "type": "any", "attributes": []}}


def scope():
    return JobScope(206, {0: FrameInfo(0, 8, 6)}, LABELS, {"rectangle", "mask"},
                    {637: "rectangle", 1750: "mask", 1751: "mask"})


def mask_shape(bitmap, label_id=1750):
    return {"frame": 0, "label_id": label_id, "type": "mask", "points": encode_mask(bitmap),
            "source": "auto", "attributes": [], "rotation": 0., "score": .9}


class SceneTests(unittest.TestCase):
    def test_scene_dry_run_routes_box_and_mask_without_writing(self):
        profile = Path(__file__).parents[1] / "scene_job206.toml"
        _, mapping = load_detection_config(profile)
        names = dict(enumerate(mapping))
        formats = {key: "mask" if name.startswith("semantic/") else "rectangle" for key, name in names.items()}
        labels = []
        for source, label_id in mapping.items():
            group, name = source.split("/", 1)
            if group == "det":
                name = {"person": "pedestrian", "motor": "motorcycle", "bike": "bicycle"}.get(name, name)
            elif name.startswith("traffic "):
                name = name.replace(" ", "_")
            labels.append({"id": label_id, "name": name, "type": "any", "attributes": []})
        client = Mock(redactor=Redactor())
        client.about.return_value = {"version": "2.76.0"}
        client.current_user.return_value = {"id": 212}
        client.job.return_value = {"id": 206, "task_id": 71, "organization": 1, "type": "annotation",
                                   "dimension": "2d", "media_type": "image", "mode": "annotation", "start_frame": 0, "stop_frame": 0}
        client.organization.return_value = {"id": 1, "slug": "cohort-5a"}
        client.job_labels.return_value = labels
        client.metadata.return_value = {"start_frame": 0, "stop_frame": 0, "size": 1, "frame_filter": "",
                                        "frames": [{"width": 8, "height": 6}], "deleted_frames": []}
        client.annotations.return_value = {"shapes": [{"id": 5, "label_id": 639, "frame": 0,
                                                       "type": "rectangle", "points": [1., 1., 3., 4.]}]}
        raw = io.BytesIO()
        Image.new("RGB", (8, 6)).save(raw, format="PNG")
        client.frame_bytes.return_value = raw.getvalue()
        detector = Mock(names=names, shape_types=formats, weights_hash="fake-scene-weights")
        road_class = next(key for key, name in names.items() if name == "semantic/road")
        detector.predict.return_value = [[Detection(2, .9, (1., 1., 3., 4.)),
                                          Detection(road_class, .9, tuple(encode_mask(np.ones((6, 8), dtype=bool))), "mask")]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            with patch("main.CvatClient", return_value=client), patch("main.read_credentials"):
                with patch("scene_detector.SceneDetector", return_value=detector):
                    with patch("main.RunState", side_effect=lambda context: RunState(context, path)):
                        output = io.StringIO()
                        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                            self.assertEqual(main(["--config", str(profile), "--limit", "1"]), 0)
            self.assertIn("mask_bbox=", output.getvalue())
            self.assertIn("New shapes prepared: 1", output.getvalue())
        client.create_shapes.assert_not_called()
        client.schema.assert_not_called()
        client.frame_bytes.assert_called_once()

    def test_detection_checkpoint_format_is_unchanged(self):
        expected = {"model": "models/yolo11n.pt", "confidence": .45, "iou": .50, "duplicate_iou": .85,
                    "batch_size": 1, "image_size": 640, "device": "cpu"}
        self.assertEqual(DetectionConfig().checkpoint_dict(), expected)
        self.assertEqual(DetectionConfig(pipeline="scene").checkpoint_dict()["pipeline"], "scene")

    def test_mask_preserves_hole_and_is_not_changed_into_bbox(self):
        bitmap = np.ones((6, 8), dtype=bool)
        bitmap[2:4, 3:5] = False
        mapper = LabelMapper({100: "semantic/road"}, list(LABELS.values()), {"semantic/road": 1750}, shape_types={100: "mask"})
        detection = Detection(100, .9, tuple(encode_mask(bitmap)), "mask")
        shape = make_shape(0, detection, LABELS[1750], mapper, scope())
        self.assertEqual(shape["type"], "mask")
        self.assertTrue(np.array_equal(decode_mask(shape["points"], image_width=8, image_height=6), bitmap))

    def test_invalid_mask_runs_bounds_and_wrong_geometry_are_blocked(self):
        shape = mask_shape(np.ones((6, 8), dtype=bool))
        scope().validate_shape(shape)
        for bad in ({**shape, "points": [0, 48, 0, 0, 8, 5]},
                    {**shape, "points": [-1, 49, 0, 0, 7, 5]},
                    {**shape, "points": [48, 0, 0, 7, 5]},
                    {**shape, "label_id": 637}):
            with self.assertRaises(ValueError):
                scope().validate_shape(bad)

    def test_semantic_mask_dedup_removes_existing_pixels_of_other_semantic_labels(self):
        road = np.zeros((6, 8), dtype=bool)
        road[:3] = True
        sidewalk = np.zeros((6, 8), dtype=bool)
        sidewalk[3:, :4] = True
        annotations = {"shapes": [{**mask_shape(road), "id": 1}, {**mask_shape(sidewalk, 1751), "id": 2}]}
        original = copy.deepcopy(annotations)
        full = mask_shape(np.ones((6, 8), dtype=bool))
        accepted, count = deduplicate([full, full], annotations, .7, scope=scope())
        self.assertEqual(len(accepted), 1)
        self.assertEqual(count, 1)
        actual = decode_mask(accepted[0]["points"], image_width=8, image_height=6)
        self.assertTrue(np.array_equal(actual, ~(road | sidewalk)))
        self.assertEqual(annotations, original)

    def test_rectangle_objects_do_not_block_different_semantic_labels(self):
        full = mask_shape(np.ones((6, 8), dtype=bool))
        existing = {"shapes": [{"label_id": 637, "frame": 0, "type": "rectangle", "points": [0, 0, 8, 6]}]}
        accepted, count = deduplicate([full], existing, .7, scope=scope())
        self.assertEqual(count, 0)
        self.assertEqual(accepted[0]["points"], full["points"])

    def test_polygon_semantic_annotations_are_protected(self):
        existing = {"shapes": [{"label_id": 1751, "frame": 0, "type": "polygon", "points": [0, 0, 3, 0, 3, 2, 0, 2]}]}
        accepted, _ = deduplicate([mask_shape(np.ones((6, 8), dtype=bool))], existing, .7, scope=scope())
        decoded = decode_mask(accepted[0]["points"], image_width=8, image_height=6)
        self.assertFalse(decoded[:3, :4].any())
        self.assertTrue(decoded[3:, 4:].all())

    def test_mask_label_type_is_checked_by_mapper(self):
        with self.assertRaises(ValueError):
            LabelMapper({100: "semantic/road"}, [{"id": 1, "name": "road", "type": "rectangle"}],
                        {"semantic/road": 1}, shape_types={100: "mask"})

    def test_schema_must_explicitly_support_mask(self):
        schema = valid_schema()
        with self.assertRaises(ValueError):
            verify_create_schema(schema, required_types={"rectangle", "mask"})
        geometry = schema["paths"]["/api/jobs/{id}/annotations/"]["patch"]["requestBody"]["content"]["application/json"]["schema"]["properties"]["shapes"]["items"]["properties"]["type"]["enum"]
        geometry.append("mask")
        self.assertTrue(verify_create_schema(schema, required_types={"rectangle", "mask"}))

    def test_sdk_serializes_mask_as_mask_and_never_as_rectangle(self):
        client = CvatClient(Config(), Redactor())
        self.addCleanup(client.close)
        request = Mock(return_value=SimpleNamespace(status=200, data=b'{"shapes":[]}', headers={}, release_conn=Mock()))
        client.sdk.rest_client.pool_manager.request = request
        with contextlib.redirect_stdout(io.StringIO()):
            client.create_shapes([mask_shape(np.ones((6, 8), dtype=bool))], scope=scope(), schema_verified=True, confirmed=True)
        payload = json.loads(request.call_args.kwargs["body"])
        self.assertEqual(request.call_args.args[0], "PATCH")
        self.assertTrue(request.call_args.args[1].endswith("/api/jobs/206/annotations/?action=create"))
        self.assertEqual(payload["shapes"][0]["type"], "mask")
        self.assertTrue(decode_mask(payload["shapes"][0]["points"], image_width=8, image_height=6).all())

    def test_append_mask_preserves_old_rectangle_and_reads_back_new_mask(self):
        client = Mock(config=Config())
        client.job.return_value = {"id": 206, "task_id": 71, "organization": 1, "type": "annotation",
                                   "dimension": "2d", "media_type": "image", "mode": "annotation", "start_frame": 0, "stop_frame": 0}
        client.organization.return_value = {"id": 1, "slug": "cohort-5a"}
        client.metadata.return_value = {"start_frame": 0, "stop_frame": 0, "size": 1, "frame_filter": "",
                                        "frames": [{"width": 8, "height": 6}], "deleted_frames": []}
        client.job_labels.return_value = list(LABELS.values())
        old = {"id": 5, "label_id": 637, "frame": 0, "type": "rectangle", "points": [1, 1, 3, 4]}
        saved = {**mask_shape(np.ones((6, 8), dtype=bool)), "id": 6}
        client.annotations.side_effect = [{"shapes": [old]}, {"shapes": [old]}, {"shapes": [old, saved]}]
        client.create_shapes.return_value = {"shapes": [saved]}
        with contextlib.redirect_stdout(io.StringIO()):
            created, _ = apply_shapes(client, [{k: v for k, v in saved.items() if k != "id"}], scope(), DetectionConfig(), confirmed=True, schema_verified=True)
        self.assertEqual(created, 1)
        self.assertEqual(old["points"], [1, 1, 3, 4])

    def test_profile_separates_detection_and_semantic_alias_ids(self):
        settings, mapping = load_detection_config(Path(__file__).parents[1] / "scene_job206.toml")
        self.assertEqual(settings.pipeline, "scene")
        self.assertEqual(mapping["det/person"], 637)
        self.assertEqual(mapping["semantic/person"], 1761)
        self.assertEqual(mapping["det/traffic light"], 645)
        self.assertEqual(mapping["semantic/traffic light"], 1756)
        self.assertEqual(len(mapping), 22)
        self.assertTrue(all(not k.startswith(("lane/", "area/")) for k in mapping))

    def test_semantic_prediction_resizes_to_original_pixels_and_keeps_holes(self):
        import torch
        detector = SceneDetector.__new__(SceneDetector)
        detector.config = DetectionConfig(semantic_min_pixels=1)
        detector.torch = torch
        detector.semantic_ids = {0: "road", 10: "sky"}
        detector.processor = Mock(return_value={})
        logits = torch.full((1, 19, 4, 5), -8.)
        logits[0, 0, :2] = 8.
        logits[0, 10, 2:] = 8.
        logits[0, 12, 1, 1] = 16.  # rider pixel creates a hole in road; not relabeled as road.
        detector.semantic = Mock(return_value=SimpleNamespace(logits=logits))
        with Image.new("RGB", (5, 4)) as image:
            predictions = detector.semantic_predictions(image)
        road = next(p for p in predictions if p.class_id == 100)
        sky = next(p for p in predictions if p.class_id == 110)
        road_mask = decode_mask(road.points, image_width=5, image_height=4)
        self.assertEqual(int(road_mask.sum()), 9)
        self.assertFalse(road_mask[1, 1])
        self.assertEqual(int(decode_mask(sky.points, image_width=5, image_height=4).sum()), 10)


if __name__ == "__main__":
    unittest.main()
