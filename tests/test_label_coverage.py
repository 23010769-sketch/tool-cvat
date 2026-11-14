"""Mức hỗ trợ labels không được nhầm với số ảnh hoặc bảo đảm phát hiện đối tượng."""

import contextlib
import io
import unittest
from unittest.mock import Mock, patch

from auth import Redactor
from config import parse_config
from label_coverage import label_coverage, print_label_coverage, require_full_coverage
from main import main


class LabelCoverageTests(unittest.TestCase):
    def test_reports_supported_missing_and_geometry_requirements(self):
        labels = [
            {"id": 1, "name": "Person", "type": "rectangle"},
            {"id": 2, "name": "Lane", "type": "polyline"},
            {"id": 3, "name": "Drivable Area", "type": "polygon"},
            {"id": 4, "name": "Traffic Sign", "type": "rectangle"},
        ]
        rows = label_coverage({0: "person", 1: "stop sign"}, labels)
        self.assertEqual([r["status"] for r in rows], ["supported", "unsupported", "unsupported", "unsupported"])
        self.assertIn("polyline", rows[1]["reason"])
        self.assertIn("segmentation", rows[2]["reason"])
        # A stop sign model class does not imply support for all traffic signs.
        self.assertEqual(rows[3]["sources"], [])

    def test_ambiguity_does_not_claim_both_labels_are_supported(self):
        labels = [{"id": 1, "name": "Person"}, {"id": 2, "name": "Pedestrian"}]
        rows = label_coverage({0: "person"}, labels)
        self.assertEqual([r["status"] for r in rows], ["ambiguous", "ambiguous"])
        chosen = label_coverage({0: "person"}, labels, {"person": 1})
        self.assertEqual([r["status"] for r in chosen], ["supported", "not_selected"])

    def test_skipped_class_and_missing_attribute_default_are_not_covered(self):
        labels = [{"id": 1, "name": "Person", "attributes": [{"id": 8}]}]
        self.assertEqual(label_coverage({0: "person"}, labels)[0]["status"], "attributes")
        self.assertEqual(label_coverage({0: "person"}, labels, {"person": ""})[0]["status"], "not_selected")

    def test_skeleton_sublabels_are_reported_without_fake_rectangle_mapping(self):
        labels = [{"id": 1, "name": "Person", "type": "skeleton",
                   "sublabels": [{"id": 2, "name": "head", "type": "points"}]}]
        rows = label_coverage({0: "person"}, labels)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status"] == "unsupported" for row in rows))
        self.assertEqual(rows[1]["name"], "Person/head")

    def test_full_coverage_guard_blocks_any_missing_label(self):
        rows = label_coverage({0: "person"}, [{"id": 1, "name": "Person"}])
        require_full_coverage(rows)
        rows.extend(label_coverage({0: "person"}, [{"id": 2, "name": "Lane", "type": "polyline"}]))
        with self.assertRaises(ValueError) as caught:
            require_full_coverage(rows)
        self.assertIn("ID: 2", str(caught.exception))

    def test_output_contains_ids_types_and_safe_names(self):
        redactor = Redactor()
        redactor.add("private-string")
        rows = label_coverage({0: "person"}, [{"id": 1, "name": "private-string", "type": "polyline"}])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            print_label_coverage(rows, redactor)
        self.assertIn("0/1", out.getvalue())
        self.assertIn("ID 1", out.getvalue())
        self.assertNotIn("private-string", out.getvalue())

    def test_audit_cannot_be_combined_with_image_or_write_modes(self):
        for options in (["--apply"], ["--all"], ["--limit", "1"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_config(["--audit-labels", *options], annotation=True)

    @patch("main.read_credentials")
    @patch("main.YoloDetector")
    @patch("main.CvatClient")
    def test_audit_uses_job_labels_only_without_prompt_or_frames_or_state(self, client_type, detector_type, credentials):
        client = client_type.return_value
        client.about.return_value = {"version": "2.76.0"}
        client.current_user.return_value = {"id": 212}
        client.job.return_value = {"id": 206, "task_id": 71, "organization": 1, "type": "annotation",
                                   "dimension": "2d", "media_type": "image"}
        client.organization.return_value = {"id": 1, "slug": "cohort-5a"}
        client.job_labels.return_value = [{"id": 1, "name": "Person"}, {"id": 2, "name": "Pedestrian"},
                                          {"id": 3, "name": "Lane", "type": "polyline"}]
        detector_type.return_value.names = {0: "person"}
        with patch("builtins.input") as prompt, patch("main.RunState") as state, patch("main.save_label_audit") as report:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--audit-labels", "--auth", "password"]), 0)
            prompt.assert_not_called()
            state.assert_not_called()
            report.assert_called_once()
        client.metadata.assert_not_called()
        client.annotations.assert_not_called()
        client.frame_bytes.assert_not_called()
        client.create_shapes.assert_not_called()
        detector_type.return_value.predict.assert_not_called()

    @patch("main.read_credentials")
    @patch("main.YoloDetector")
    @patch("main.CvatClient")
    def test_require_all_blocks_frame_inference_when_lane_is_missing(self, client_type, detector_type, credentials):
        client = client_type.return_value
        client.about.return_value = {"version": "2.76.0"}
        client.current_user.return_value = {"id": 212}
        client.job.return_value = {"id": 206, "task_id": 71, "organization": 1, "type": "annotation",
                                   "dimension": "2d", "media_type": "image", "start_frame": 0, "stop_frame": 0, "mode": "annotation"}
        client.organization.return_value = {"id": 1, "slug": "cohort-5a"}
        client.job_labels.return_value = [{"id": 1, "name": "Person"}, {"id": 2, "name": "Lane", "type": "polyline"}]
        client.metadata.return_value = {"start_frame": 0, "stop_frame": 0, "frame_filter": "", "size": 1,
                                        "frames": [{"width": 100, "height": 80}], "deleted_frames": []}
        detector_type.return_value.names = {0: "person"}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--all", "--apply", "--require-all-labels"]), 1)
        client.frame_bytes.assert_not_called()
        client.create_shapes.assert_not_called()
        detector_type.return_value.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
