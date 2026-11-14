"""Box geometry, reference matching and explicit conversion with mock Job APIs."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from cvat_sdk.masks import encode_mask

from annotation_writer import make_shape, verify_patch_schema, same_polyline
from auth import Redactor
from bdd_reference import (BDD_NAME, BOX_NAMES, BddReferenceDetector, BoxSceneDetector,
                           PublicReferenceStore, clip_polyline, convert_objects, flatten_curve, frame_names)
from box_conversion import (apply_conversions, geometry_boxes, plan_conversions, run_conversion, validate_update)
from config import Config, DetectionConfig, load_detection_config
from cvat_client import CvatClient
from frame_mapping import FrameInfo, JobScope
from label_mapper import LabelMapper
from main import main
from run_state import RunState
from scene_detector import SceneDetector
from yolo_detector import Detection
from test_annotation import valid_schema


ROOT = Path(__file__).parents[1]
STEM = "b1c66a42-6f7d68ca"
JOB = {"id": 206, "task_id": 71, "organization": 1, "type": "annotation", "dimension": "2d",
       "media_type": "image", "mode": "annotation", "start_frame": 0, "stop_frame": 0}
META = {"start_frame": 0, "stop_frame": 0, "size": 1, "frame_filter": "", "deleted_frames": [],
        "frames": [{"width": 12, "height": 10, "name": STEM + ".jpg"}]}
LABELS = {639: {"id": 639, "name": "car", "type": "any", "attributes": []},
          1755: {"id": 1755, "name": "pole", "type": "any", "attributes": []}}


def scope():
    return JobScope(206, {0: FrameInfo(0, 12, 10)}, copy.deepcopy(LABELS), {"rectangle"}, {}, {"auto", "file"})


def old_mask():
    bitmap = np.zeros((10, 12), dtype=bool)
    bitmap[1:7, 1:3] = True
    bitmap[2:6, 8:10] = True
    return {"id": 25, "frame": 0, "label_id": 1755, "type": "mask", "points": encode_mask(bitmap),
            "source": "auto", "rotation": 0., "attributes": [], "group": 0, "occluded": False, "z_order": 0}


def fake_client():
    client = Mock(config=Config(), redactor=Redactor())
    client.about.return_value = {"version": "2.76.0"}
    client.current_user.return_value = {"id": 212}
    client.job.return_value = copy.deepcopy(JOB)
    client.organization.return_value = {"id": 1, "slug": "cohort-5a"}
    client.metadata.return_value = copy.deepcopy(META)
    client.job_labels.return_value = list(LABELS.values())
    return client


class BoxTests(unittest.TestCase):
    def test_semantic_boxes_separate_disconnected_objects_and_filter_noise(self):
        import torch
        detector = SceneDetector.__new__(SceneDetector)
        detector.config = DetectionConfig(output_geometry="rectangle", semantic_min_pixels=3)
        detector.torch = torch
        detector.semantic_ids = {5: "pole"}
        detector.processor = Mock(return_value={})
        logits = torch.full((1, 19, 10, 12), -9.)
        logits[0, 18] = 8.
        logits[0, 5, 1:7, 1:3] = 20.
        logits[0, 5, 2:6, 8:10] = 20.
        logits[0, 5, 9, 0] = 20.  # single noisy pixel is not a pole box
        detector.semantic = Mock(return_value=SimpleNamespace(logits=logits))
        with Image.new("RGB", (12, 10)) as image:
            output = detector.semantic_predictions(image)
        self.assertEqual([d.points for d in output], [(1., 1., 3., 7.), (8., 2., 10., 6.)])
        self.assertTrue(all(d.geometry == "rectangle" and d.source == "auto" for d in output))

    def test_scene_profile_now_requests_rectangles(self):
        settings, _ = load_detection_config(ROOT / "scene_job206.toml")
        self.assertEqual(settings.output_geometry, "rectangle")

    def test_combined_profile_has_31_distinct_label_ids_all_rectangles(self):
        settings, mapping = load_detection_config(ROOT / "all_boxes_job206.toml")
        source = BoxSceneDetector.__new__(BoxSceneDetector)
        reference = BddReferenceDetector(settings)
        scene_names = {key: "det/" + name for key, name in enumerate(
            ("person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign"))}
        semantic_names = ("road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                          "traffic sign", "vegetation", "terrain", "sky", "person")
        scene_names.update({100 + key: "semantic/" + name for key, name in enumerate(semantic_names)})
        with patch("scene_detector.SceneDetector", return_value=Mock(names=scene_names, weights_hash="model")):
            source = BoxSceneDetector(settings)
        self.assertEqual(len(source.names), 31)
        self.assertEqual(set(source.shape_types.values()), {"rectangle"})
        labels = [{"id": label_id, "name": name, "type": "any", "attributes": []} for name, label_id in mapping.items()]
        mapper = LabelMapper(source.names, labels, mapping, shape_types=source.shape_types)
        self.assertEqual({label["id"] for label in mapper.mapping.values()}, set(range(637, 656)) | set(range(1750, 1762)))
        self.assertEqual(len(reference.names), 10)

    def test_reference_curves_and_area_become_boxes_without_mislabeling(self):
        data = {"frames": [{"objects": [
            {"category": "lane/single yellow", "poly2d": [[4, 1, "L"], [4, 9, "L"]]},
            {"category": "area/drivable", "poly2d": [[0, 5, "L"], [12, 5, "L"], [12, 10, "L"], [0, 10, "L"]]},
            {"category": "traffic sign", "box2d": {"x1": 5, "y1": 1, "x2": 7, "y2": 3}},
            {"category": "lane/double other", "poly2d": [[0, 0, "L"], [1, 1, "L"]]},
        ]}]}
        detections, ignored = convert_objects(data, 12, 10, boxes=True)
        self.assertEqual([BOX_NAMES[d.class_id] for d in detections], ["lane/single yellow", "area/drivable", "traffic sign"])
        self.assertEqual(detections[0].points, (3.5, 1., 4.5, 9.))
        self.assertTrue(all(d.geometry == "rectangle" and d.source == "file" for d in detections))
        self.assertEqual(ignored["lane/double other"], 1)

    def test_bezier_is_evaluated_not_treated_as_control_polygon(self):
        vertices = flatten_curve([[0, 0, "L"], [0, 10, "C"], [10, 10, "C"], [10, 0, "C"]])
        self.assertGreater(len(vertices), 4)
        self.assertAlmostEqual(max(y for _, y in vertices), 7.5)
        with self.assertRaises(ValueError):
            flatten_curve([[0, 0, "L"], [2, 8, "C"], [10, 0, "C"]])

    def test_file_source_is_explicit_and_never_claims_model_confidence(self):
        target = LABELS[1755]
        mapper = LabelMapper({1: "pole"}, [target])
        result = make_shape(0, Detection(1, 1., (1, 1, 3, 7), source="file"), target, mapper, scope())
        self.assertEqual(result["source"], "file")
        self.assertNotIn("score", result)
        restricted = scope()
        restricted.allowed_sources = {"auto"}
        with self.assertRaises(ValueError):
            restricted.validate_shape(result)

    def test_reference_frame_names_use_job_relative_metadata_and_reject_renamed_images(self):
        job = {**JOB, "start_frame": 10, "stop_frame": 12}
        meta = {"frames": [{"name": "folder/" + STEM + ".jpg"}, {"name": "placeholder.jpg"},
                            {"name": "abcdef12-12345678.jpg"}]}
        self.assertEqual(frame_names(job, meta, {10: None, 12: None}), {10: STEM, 12: "abcdef12-12345678"})
        with self.assertRaises(ValueError):
            frame_names(JOB, {"frames": [{"name": "image1.jpg"}]}, {0: None})
        store = PublicReferenceStore()
        self.addCleanup(store.close)
        store.session.get = Mock()
        with self.assertRaises(ValueError):
            store.record("../../private")
        store.session.get.assert_not_called()

    def test_all_reference_frames_must_match_rgb_before_they_are_exposed(self):
        settings = DetectionConfig(pipeline="bdd-reference", output_geometry="rectangle")
        detector = BddReferenceDetector(settings)
        raw = io.BytesIO()
        Image.new("RGB", (12, 10), "red").save(raw, format="PNG")
        data = {"name": STEM, "frames": [{"objects": []}]}
        store = Mock()
        store.record.return_value = (data, "public-image.jpg", "json-sha")
        store.get.return_value = raw.getvalue()
        client = fake_client()
        client.frame_bytes.return_value = raw.getvalue()
        with patch("bdd_reference.PublicReferenceStore", return_value=store), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            detector.prepare(client, JOB, META, scope())
        self.assertEqual(detector.for_frames([0]), [[]])
        client.create_shapes.assert_not_called()
        wrong = io.BytesIO()
        Image.new("RGB", (12, 10), "blue").save(wrong, format="PNG")
        client.frame_bytes.return_value = wrong.getvalue()
        with patch("bdd_reference.PublicReferenceStore", return_value=store), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "KHÔNG trùng"):
                detector.prepare(client, JOB, META, scope())
        with self.assertRaises(ValueError):
            detector.for_frames([0])

    def test_native_polyline_clipping_and_dedup_keep_distinct_crossing_lines(self):
        self.assertEqual(clip_polyline([(-2, 2), (8, 2)], 5, 5), [[(0., 2.), (5., 2.)]])
        self.assertTrue(same_polyline([0, 0, 10, 10], [10, 10, 5, 5, 0, 0]))
        self.assertFalse(same_polyline([0, 0, 10, 10], [0, 10, 10, 0]))


class ConversionTests(unittest.TestCase):
    def test_mask_components_get_separate_boxes_and_keep_original_id(self):
        old = old_mask()
        car = {"id": 1, "frame": 0, "label_id": 639, "type": "rectangle", "points": [0, 0, 2, 2]}
        manual = {**old, "id": 26, "source": "manual"}
        annotations = {"shapes": [car, old, manual]}
        before = copy.deepcopy(annotations)
        updates, additions, originals, skipped = plan_conversions(annotations, scope(), [0], 1)
        self.assertEqual(updates[0]["id"], 25)
        self.assertEqual(updates[0]["points"], [1., 1., 3., 7.])
        self.assertEqual(additions[0]["points"], [8., 2., 10., 6.])
        self.assertNotIn("id", additions[0])
        self.assertEqual(skipped, 1)
        self.assertEqual(annotations, before)
        self.assertEqual(originals[25], old)

    def test_conversion_refuses_label_or_frame_changes_and_wrong_geometry_schema(self):
        old = old_mask()
        updates, _, _, _ = plan_conversions({"shapes": [old]}, scope(), [0], 1)
        for wrong in ({**updates[0], "label_id": 639}, {**updates[0], "frame": 1}, {**updates[0], "id": 99}):
            with self.assertRaises(ValueError):
                validate_update(wrong, old, scope())
        wrong_scope = scope()
        wrong_scope.labels[1755]["type"] = "mask"
        with self.assertRaises(ValueError):
            validate_update(updates[0], old, wrong_scope)

    def test_corrupt_masks_and_rotation_fail_before_conversion(self):
        for wrong in ({**old_mask(), "points": [0, 5, 0, 0, 11, 9]}, {**old_mask(), "rotation": 20}):
            with self.assertRaises(ValueError):
                geometry_boxes(wrong, FrameInfo(0, 12, 10), 1)

    def test_dry_conversion_main_never_loads_models_downloads_frames_or_writes(self):
        client = fake_client()
        client.annotations.return_value = {"shapes": [old_mask()]}
        with patch("main.CvatClient", return_value=client), patch("main.read_credentials"), patch("main.load_detector") as model:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--convert-existing-to-boxes", "--all"]), 0)
        model.assert_not_called()
        client.frame_bytes.assert_not_called()
        client.create_shapes.assert_not_called()
        client.update_shape_geometry.assert_not_called()
        client.schema.assert_not_called()

    def test_sdk_conversion_sends_update_with_original_id_and_no_other_objects(self):
        old = old_mask()
        updates, _, originals, _ = plan_conversions({"shapes": [old]}, scope(), [0], 1)
        client = CvatClient(Config(), Redactor())
        self.addCleanup(client.close)
        request = Mock(return_value=SimpleNamespace(status=200, data=b'{"shapes":[]}', headers={}, release_conn=Mock()))
        client.sdk.rest_client.pool_manager.request = request
        with contextlib.redirect_stdout(io.StringIO()):
            client.update_shape_geometry(updates, originals=originals, scope=scope(), schema_verified=True, confirmed=True)
        self.assertEqual(request.call_args.args[:2], ("PATCH", "https://cvat.edufun.ai/api/jobs/206/annotations/?action=update"))
        payload = json.loads(request.call_args.kwargs["body"])
        self.assertEqual(payload["shapes"][0]["id"], 25)
        self.assertEqual(payload["shapes"][0]["type"], "rectangle")
        self.assertEqual(payload["tracks"], [])
        self.assertEqual(payload["tags"], [])
        with self.assertRaises(ValueError):
            client.update_shape_geometry(updates, originals=originals, scope=scope(), schema_verified=True, confirmed=False)

    def test_conversion_reads_back_original_id_and_preserves_car(self):
        old = old_mask()
        updates, _, originals, _ = plan_conversions({"shapes": [old]}, scope(), [0], 1)
        car = {"id": 1, "label_id": 639, "frame": 0, "type": "rectangle", "points": [0, 0, 2, 2]}
        client = fake_client()
        client.annotations.side_effect = [{"shapes": [old, car]}, {"shapes": [old, car]}, {"shapes": [updates[0], car]}]
        client.update_shape_geometry.return_value = {"shapes": updates}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(apply_conversions(client, scope(), updates, [], originals, DetectionConfig(), confirmed=True, schema_verified=True), 1)
        client.update_shape_geometry.assert_called_once()
        client.create_shapes.assert_not_called()

    def test_changed_original_or_failed_component_save_stops_before_geometry_update(self):
        old = old_mask()
        updates, additions, originals, _ = plan_conversions({"shapes": [old]}, scope(), [0], 1)
        client = fake_client()
        client.annotations.return_value = {"shapes": [{**old, "source": "manual"}]}
        with self.assertRaises(ValueError):
            apply_conversions(client, scope(), updates, additions, originals, DetectionConfig(), confirmed=True, schema_verified=True)
        client.update_shape_geometry.assert_not_called()
        client.annotations.return_value = {"shapes": [old]}
        with patch("box_conversion.apply_shapes", side_effect=ValueError("create failed")):
            with self.assertRaises(ValueError):
                apply_conversions(client, scope(), updates, additions, originals, DetectionConfig(), confirmed=True, schema_verified=True)
        client.update_shape_geometry.assert_not_called()

    def test_conversion_requires_confirmation_before_creating_components(self):
        client = fake_client()
        updates, additions, originals, _ = plan_conversions({"shapes": [old_mask()]}, scope(), [0], 1)
        with self.assertRaises(ValueError):
            apply_conversions(client, scope(), updates, additions, originals, DetectionConfig())
        client.annotations.assert_not_called()
        client.create_shapes.assert_not_called()
        client.update_shape_geometry.assert_not_called()

    def test_update_schema_requires_update_action_and_annotation_id(self):
        schema = valid_schema()
        with self.assertRaises(ValueError):
            verify_patch_schema(schema, action="update")
        properties = schema["paths"]["/api/jobs/{id}/annotations/"]["patch"]["requestBody"]["content"]["application/json"]["schema"]["properties"]["shapes"]["items"]["properties"]
        properties["id"] = {}
        self.assertTrue(verify_patch_schema(schema, action="update"))
        with self.assertRaises(ValueError):
            verify_patch_schema(schema, action="delete")

    def test_server_source_string_schema_accepts_known_file_source(self):
        schema = valid_schema()
        properties = schema["paths"]["/api/jobs/{id}/annotations/"]["patch"]["requestBody"]["content"]["application/json"]["schema"]["properties"]["shapes"]["items"]["properties"]
        properties["source"] = {"type": "string", "minLength": 1, "default": "manual"}
        self.assertTrue(verify_patch_schema(schema, action="create", required_sources={"auto", "file"}))
        properties["source"]["enum"] = ["auto"]
        with self.assertRaises(ValueError):
            verify_patch_schema(schema, action="create", required_sources={"file"})

    def test_staged_conversion_saves_components_before_changing_mask_and_is_idempotent(self):
        client = fake_client()
        old = old_mask()
        car = {"id": 1, "label_id": 639, "frame": 0, "type": "rectangle", "points": [0, 0, 2, 2]}
        state = {"shapes": [copy.deepcopy(old), copy.deepcopy(car)], "tracks": [], "tags": [], "intervals": []}
        order = []
        client.annotations.side_effect = lambda: copy.deepcopy(state)

        def create(shapes, **kwargs):
            order.append("create")
            saved = [{**copy.deepcopy(shape), "id": 100 + k} for k, shape in enumerate(shapes)]
            state["shapes"].extend(saved)
            return {"shapes": saved}

        def update(shapes, **kwargs):
            order.append("update")
            for shape in shapes:
                state["shapes"] = [copy.deepcopy(shape) if s["id"] == shape["id"] else s for s in state["shapes"]]
            return {"shapes": copy.deepcopy(shapes)}

        client.create_shapes.side_effect = create
        client.update_shape_geometry.side_effect = update
        schema = valid_schema()
        schema["paths"]["/api/jobs/{id}/annotations/"]["patch"]["requestBody"]["content"]["application/json"]["schema"]["properties"]["shapes"]["items"]["properties"]["id"] = {}
        client.schema.return_value = schema
        args = SimpleNamespace(all=True, apply=True, limit=None)
        with patch("box_conversion.sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="CONVERT JOB 206 TO BOXES"):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_conversion(client, scope(), JOB, DetectionConfig(semantic_min_pixels=1), args), 0)
                self.assertEqual(run_conversion(client, scope(), JOB, DetectionConfig(semantic_min_pixels=1), args), 0)
        self.assertEqual(order, ["create", "update"])
        self.assertEqual(len(state["shapes"]), 3)
        self.assertTrue(all(s["type"] == "rectangle" for s in state["shapes"]))
        self.assertEqual(next(s for s in state["shapes"] if s["id"] == 1), car)
        self.assertEqual(next(s for s in state["shapes"] if s["id"] == 25)["label_id"], 1755)

    def test_conversion_cancel_does_not_send_any_write(self):
        client = fake_client()
        client.annotations.return_value = {"shapes": [old_mask()]}
        schema = valid_schema()
        schema["paths"]["/api/jobs/{id}/annotations/"]["patch"]["requestBody"]["content"]["application/json"]["schema"]["properties"]["shapes"]["items"]["properties"]["id"] = {}
        client.schema.return_value = schema
        with patch("box_conversion.sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="NO"):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_conversion(client, scope(), JOB, DetectionConfig(), SimpleNamespace(all=True, apply=True, limit=None)), 2)
        client.create_shapes.assert_not_called()
        client.update_shape_geometry.assert_not_called()


if __name__ == "__main__":
    unittest.main()
