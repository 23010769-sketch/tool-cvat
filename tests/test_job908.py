"""Semantic entry point and compatibility of the older scoped model pipeline."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from annotate_job908 import main as job908_main
from auth import Redactor
from config import load_detection_config, parse_config
from label_coverage import label_coverage
from label_mapper import AmbiguousMappingError, LabelMapper
from main import main
from run_state import RunState
from scene_detector import SEMANTIC_NAMES
from yolo_detector import Detection


ROOT = Path(__file__).parents[1]
DETECTION_NAMES = ("person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign")
NAMES = {**{key: "det/" + name for key, name in enumerate(DETECTION_NAMES)},
         **{100 + key: "semantic/" + name for key, name in enumerate(SEMANTIC_NAMES)}}
FORMATS = {key: "rectangle" for key in NAMES}


class Job908Tests(unittest.TestCase):
    def test_entry_point_defaults_to_reviewed_semantic_preview(self):
        with patch("annotate_job908.annotation_main", return_value=0) as runner:
            self.assertEqual(job908_main([]), 0)
        self.assertEqual(runner.call_args.args[0], ["--auth", "password", "--frame-timeout", "180"])

    def test_entry_point_apply_is_explicit_and_cannot_target_other_jobs(self):
        with patch("annotate_job908.annotation_main", return_value=0) as runner:
            self.assertEqual(job908_main(["--apply", "--backend", "rest"]), 0)
        self.assertEqual(runner.call_args.args[0], ["--auth", "password", "--frame-timeout", "180", "--apply"])
        for invalid in (["--job-id", "206"], ["--task-id", "71"], ["--audit-labels", "--apply"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                job908_main(invalid)

    def test_previous_redo_flag_is_an_alias_for_default(self):
        with patch("annotate_job908.annotation_main", return_value=0) as runner:
            self.assertEqual(job908_main(["--redo-guidelines", "--apply"]), 0)
        self.assertEqual(runner.call_args.args[0], ["--auth", "password", "--frame-timeout", "180", "--apply"])

    def test_namespaced_mapping_uses_live_ids_and_separates_semantic_aliases(self):
        target_names = ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
                        "traffic light", "traffic sign", "road", "sidewalk", "building", "wall", "fence", "pole",
                        "traffic_light", "traffic_sign", "vegetation", "terrain", "sky", "person")
        labels = [{"id": 12000 + index, "name": name, "type": "any", "attributes": []}
                  for index, name in enumerate(target_names)]
        ids = {label["name"]: label["id"] for label in labels}
        mapper = LabelMapper(NAMES, labels, shape_types=FORMATS)
        self.assertEqual(len(mapper.mapping), 22)
        self.assertEqual(mapper.label(0)["id"], ids["pedestrian"])
        self.assertEqual(mapper.label(111)["id"], ids["person"])
        self.assertEqual(mapper.label(9)["id"], ids["traffic sign"])
        self.assertEqual(mapper.label(107)["id"], ids["traffic_sign"])
        self.assertEqual(mapper.label(8)["id"], ids["traffic light"])
        self.assertEqual(mapper.label(106)["id"], ids["traffic_light"])
        self.assertTrue(all(row["status"] == "supported" for row in label_coverage(NAMES, labels, shape_types=FORMATS)))

    def test_partial_job_schema_skips_missing_classes_and_duplicate_sources(self):
        labels = [{"id": 20000 + index, "name": name, "type": "rectangle", "attributes": []}
                  for index, name in enumerate(("person", "traffic_sign", "road", "car"))]
        mapper = LabelMapper(NAMES, labels, shape_types=FORMATS)
        self.assertEqual(len(mapper.mapping), 4)
        self.assertIsNone(mapper.label(111))
        self.assertIsNone(mapper.label(107))
        self.assertEqual(mapper.label(0)["id"], 20000)
        self.assertEqual(mapper.label(9)["id"], 20001)

    def test_repeated_live_names_still_require_selection(self):
        labels = [{"id": 4, "name": "Car", "type": "any"}, {"id": 5, "name": "car", "type": "any"}]
        with self.assertRaises(AmbiguousMappingError):
            LabelMapper({2: "det/car"}, labels)

    def test_explicit_string_mapping_distinguishes_spaces_and_underscores(self):
        labels = [{"id": 1, "name": "traffic sign", "type": "any"}, {"id": 2, "name": "traffic_sign", "type": "any"}]
        mapper = LabelMapper({0: "sign"}, labels, {"sign": "traffic_sign"})
        self.assertEqual(mapper.label(0)["id"], 2)

    def test_box_only_model_does_not_map_polygon_labels_or_unsupported_lanes(self):
        labels = [{"id": 1, "name": "road", "type": "polygon"},
                  {"id": 2, "name": "lane/single white", "type": "any"},
                  {"id": 3, "name": "car", "type": "any"}]
        mapper = LabelMapper(NAMES, labels, shape_types=FORMATS)
        self.assertEqual({target["id"] for target in mapper.mapping.values()}, {3})
        rows = label_coverage(NAMES, labels, shape_types=FORMATS)
        self.assertEqual([row["status"] for row in rows], ["unsupported", "unsupported", "supported"])

    def test_new_job_dry_run_uses_nonzero_frame_ids_live_labels_and_separate_checkpoint(self):
        client = Mock(redactor=Redactor())
        client.about.return_value = {"version": "2.76.0"}
        client.current_user.return_value = {"id": 212}
        client.job.return_value = {"id": 908, "task_id": 211, "organization": 1, "type": "annotation", "dimension": "2d",
                                   "media_type": "image", "mode": "annotation", "start_frame": 80, "stop_frame": 82}
        client.organization.return_value = {"id": 1, "slug": "cohort-5a"}
        client.job_labels.return_value = [{"id": 22501, "name": "car", "type": "any", "attributes": []},
                                         {"id": 22502, "name": "road", "type": "any", "attributes": []}]
        client.metadata.return_value = {"start_frame": 80, "stop_frame": 82, "size": 3, "frame_filter": "", "deleted_frames": [],
                                        "frames": [{"name": f"week1-seg/G04/G04_S{n:03d}.jpg", "width": 100, "height": 80}
                                                   for n in (81, 82, 83)]}
        client.annotations.return_value = {"shapes": [], "tracks": []}
        buffer = io.BytesIO()
        with Image.new("RGB", (100, 80)) as image:
            image.save(buffer, format="PNG")
        client.frame_bytes.return_value = buffer.getvalue()
        detector = Mock(names=NAMES, shape_types=FORMATS, weights_hash="mock-cpu-box-model")
        detector.predict.side_effect = lambda images: [[Detection(2, .9, (10., 20., 40., 60.)),
                                                       Detection(100, .8, (0., 40., 100., 80.))] for image in images]
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "checkpoint.json"
            prior = RunState({"job": 206, "task": 71}, state_path)
            prior.record(None, True, 20)
            with patch("main.CvatClient", return_value=client), patch("main.read_credentials"), patch("main.load_detector", return_value=detector):
                with patch("main.RunState", side_effect=lambda context: RunState(context, state_path)):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(main(["--url", "https://cvat.edufun.ai", "--organization", "cohort-5a",
                                               "--task-id", "211", "--job-id", "908", "--auth", "password",
                                               "--config", str(ROOT / "scene_job908.toml"), "--all"]), 0)
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(saved[prior.key], {"apply_all": 20})
            current = next(value for key, value in saved.items() if key != prior.key)
            self.assertEqual(current, {"dry_1": 1, "dry_5": 3, "dry_all": 3})
        self.assertIn("Task ID: 211\nJob ID: 908", output.getvalue())
        self.assertIn("label_id=22501", output.getvalue())
        self.assertIn("label_id=22502", output.getvalue())
        self.assertIn("toàn Job: 3 frame", output.getvalue())
        self.assertEqual({call.args[0] for call in client.frame_bytes.call_args_list}, {80, 81, 82})
        self.assertEqual(client.frame_bytes.call_count, 7)
        client.task.assert_not_called()
        client.create_shapes.assert_not_called()
        client.update_shape_geometry.assert_not_called()
        detector.prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
