"""Kiểm kê label khác với suy luận hay ghi annotation."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from annotation_inventory import inspect_annotations, summarize_annotations
from auth import Redactor
from check_api import main as check_main
from config import Config, parse_config


LABELS = [{"id": 1, "name": "car", "type": "rectangle"},
          {"id": 2, "name": "road", "type": "mask"},
          {"id": 3, "name": "lane/single white", "type": "polyline"}]
JOB = {"id": 206, "task_id": 71, "organization": 1, "type": "annotation",
       "dimension": "2d", "media_type": "image"}


class AnnotationInventoryTests(unittest.TestCase):
    def test_label_list_alone_does_not_imply_annotations(self):
        rows = summarize_annotations(LABELS, {"shapes": [], "tracks": [], "tags": []})
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(not row["has_annotations"] for row in rows))

    def test_counts_types_and_tracks_without_counting_keyframes_as_objects(self):
        annotations = {"shapes": [{"label_id": 1, "type": "rectangle"}, {"label_id": 1, "type": "rectangle"},
                                   {"label_id": 2, "type": "mask"}],
                       "tracks": [{"label_id": 3, "shapes": [{"frame": 0}, {"frame": 10}]}]}
        rows = {row["id"]: row for row in summarize_annotations(LABELS, annotations)}
        self.assertEqual(rows[1]["shape_types"], {"rectangle": 2})
        self.assertEqual(rows[2]["shape_types"], {"mask": 1})
        self.assertEqual(rows[3]["tracks"], 1)
        self.assertEqual(rows[3]["track_keyframes"], 2)
        self.assertEqual(rows[3]["shapes"], 0)

    def test_skeleton_elements_and_unlisted_ids_are_not_silently_dropped(self):
        labels = [{"id": 1, "name": "body", "type": "skeleton", "sublabels": [{"id": 2, "name": "head", "type": "points"}]}]
        annotations = {"shapes": [{"label_id": 1, "type": "skeleton", "elements": [{"label_id": 2, "type": "points"}]},
                                   {"label_id": 99, "type": "polygon"}]}
        rows = {row["id"]: row for row in summarize_annotations(labels, annotations)}
        self.assertTrue(rows[1]["has_annotations"])
        self.assertEqual(rows[2]["shapes"], 1)
        self.assertEqual(rows[99]["type"], "unknown")

    def test_report_has_counts_but_no_coordinates_or_sensitive_names(self):
        client = Mock()
        client.job.return_value = JOB
        client.organization.return_value = {"id": 1, "slug": "cohort-5a"}
        client.job_labels.return_value = [{"id": 1, "name": "private-value", "type": "rectangle"}]
        client.annotations.return_value = {"shapes": [{"id": 500, "label_id": 1, "frame": 0,
                                                       "type": "rectangle", "points": [1, 2, 30, 40]}]}
        redactor = Redactor()
        redactor.add("private-value")
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = inspect_annotations(client, Config(), redactor, directory=directory)
            text = path.read_text(encoding="utf-8")
            report = json.loads(text)
        self.assertEqual(report["labels"][0]["shapes"], 1)
        for hidden in ("points", "private-value", '"frame"', '"id": 500'):
            self.assertNotIn(hidden, text)
        self.assertFalse(report["contains_coordinates"])
        client.metadata.assert_not_called()
        client.frame_bytes.assert_not_called()
        client.create_shapes.assert_not_called()

    def test_wrong_job_stops_before_annotation_read(self):
        client = Mock()
        client.job.return_value = {**JOB, "task_id": 72}
        client.organization.return_value = {"id": 1, "slug": "cohort-5a"}
        with self.assertRaises(ValueError):
            inspect_annotations(client, Config(), Redactor())
        client.job_labels.assert_not_called()
        client.annotations.assert_not_called()

    def test_inspection_excludes_apply_or_image_processing_flags(self):
        for extra in (["--apply"], ["--all"], ["--limit", "1"], ["--audit-labels"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_config(["--inspect-annotations", *extra], annotation=True)

    @patch("check_api.read_credentials")
    @patch("check_api.CvatClient")
    def test_check_api_inventory_uses_job_scope_without_task_access(self, client_type, credentials):
        client = client_type.return_value
        client.about.return_value = {"version": "2.76.0"}
        client.current_user.return_value = {"id": 212}
        with patch("annotation_inventory.inspect_annotations") as inventory:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check_main(["--inspect-annotations"]), 0)
            inventory.assert_called_once()
        client.task.assert_not_called()
        client.frame_bytes.assert_not_called()
        client.create_shapes.assert_not_called()


if __name__ == "__main__":
    unittest.main()
