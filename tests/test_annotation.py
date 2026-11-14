"""Kiểm tra frame mapping, dry-run và append-only bằng dữ liệu giả."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from annotation_writer import (apply_shapes, deduplicate, existing_boxes, make_shape,
                               verify_create_schema, verify_preserved, annotation_snapshot)
from auth import Redactor
from config import Config, DetectionConfig, load_detection_config, parse_config
from cvat_client import CvatClient
from frame_mapping import FrameInfo, JobScope, build_frames, validate_job
from label_mapper import AmbiguousMappingError, LabelMapper
from main import AnnotationSession, StageResult, main, print_next_action, process_frames, resolve_label_mapping
from run_state import RunState
from yolo_detector import Detection


LABELS = [{"id": 7, "name": "Pedestrian", "type": "rectangle", "attributes": []}]
JOB = {"id": 206, "task_id": 71, "organization": 1, "type": "annotation", "dimension": "2d",
       "media_type": "image", "mode": "annotation", "start_frame": 10, "stop_frame": 11}
META = {"start_frame": 130, "stop_frame": 133, "frame_filter": "step=3", "size": 2,
        "included_frames": None, "deleted_frames": [],
        "frames": [{"width": 100, "height": 80, "name": "fake.png"}] * 2}
ORG = {"id": 1, "slug": "cohort-5a"}


def valid_schema():
    return {"info": {"version": "2.76.0 (2.0)"}, "paths": {"/api/jobs/{id}/annotations/": {"patch": {
        "operationId": "jobs_partial_update_annotations",
        "parameters": [{"name": "action", "in": "query", "required": True,
                        "schema": {"enum": ["create", "update", "delete"]}}],
        "requestBody": {"content": {"application/json": {"schema": {
            "properties": {"shapes": {"type": "array", "items": {"properties": {
                "frame": {}, "label_id": {}, "points": {}, "source": {},
                "type": {"enum": ["rectangle", "polygon"]}}}}}}}}}}}}}


def make_scope():
    return JobScope(206, build_frames(JOB, META), {7: LABELS[0]})


def box(frame=10):
    mapper = LabelMapper({0: "person"}, LABELS)
    return make_shape(frame, Detection(0, .9, (10., 10., 30., 40.)), LABELS[0], mapper, make_scope())


class FrameTests(unittest.TestCase):
    def test_nonzero_start_and_step_use_task_relative_frame_ids(self):
        frames = build_frames(JOB, META)
        self.assertEqual(list(frames), [10, 11])
        self.assertEqual(frames[10], FrameInfo(10, 100, 80))

    def test_sparse_absolute_included_frames_and_relative_deletions(self):
        job = {**JOB, "stop_frame": 16}
        meta = {**META, "stop_frame": 148, "size": 3, "included_frames": [130, 139, 148],
                "deleted_frames": [13], "frames": META["frames"][:1] * 7}
        self.assertEqual(list(build_frames(job, meta)), [10, 16])

    def test_video_reuses_single_dimensions_without_changing_numbers(self):
        self.assertEqual(list(build_frames({**JOB, "mode": "interpolation"},
                                          {**META, "frames": META["frames"][:1]})), [10, 11])

    def test_unaligned_sparse_frame_stops(self):
        with self.assertRaises(ValueError):
            build_frames(JOB, {**META, "included_frames": [131], "size": 1})

    def test_ambiguous_dimensions_and_size_stop(self):
        for meta in ({**META, "frames": []}, {**META, "size": 3}, {**META, "stop_frame": 130}):
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                build_frames(JOB, meta)

    def test_validates_actual_job_workspace_and_dimensions(self):
        config = Config(organization="cohort-5a")
        validate_job(config, JOB, ORG)
        for job in ({**JOB, "id": 207}, {**JOB, "task_id": 72}, {**JOB, "organization": 2},
                    {**JOB, "dimension": "3d"}, {**JOB, "media_type": "audio"}):
            with self.assertRaises(ValueError):
                validate_job(config, job, ORG)


class MappingTests(unittest.TestCase):
    def test_ambiguity_reports_candidate_names_and_ids(self):
        labels = LABELS + [{**LABELS[0], "id": 8, "name": "Person"}]
        with self.assertRaises(AmbiguousMappingError) as caught:
            LabelMapper({0: "person"}, labels)
        self.assertIn("ID 7", str(caught.exception))
        self.assertIn("Pedestrian", str(caught.exception))
        self.assertIn("ID 8", str(caught.exception))
        self.assertIn("Person", str(caught.exception))

    def test_id_override_resolves_duplicate_normalized_names(self):
        labels = [{**LABELS[0], "name": "Person"}, {**LABELS[0], "id": 8, "name": "person"}]
        with self.assertRaises(AmbiguousMappingError):
            LabelMapper({0: "person"}, labels, {"person": "Person"})
        self.assertEqual(LabelMapper({0: "person"}, labels, {"person": 8}).label(0)["id"], 8)
        with self.assertRaises(ValueError):
            LabelMapper({0: "person"}, labels, {"person": 999})

    def test_repeated_same_id_is_not_ambiguous(self):
        self.assertEqual(LabelMapper({0: "person"}, LABELS + LABELS).label(0)["id"], 7)

    def test_interactive_mapping_requires_selection_and_validates_id(self):
        labels = LABELS + [{**LABELS[0], "id": 8, "name": "Person"}]
        overrides = {}
        output = io.StringIO()
        with patch("main.sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=["999", "8"]):
            with contextlib.redirect_stdout(output):
                mapper = resolve_label_mapping({0: "person"}, labels, overrides, Redactor())
        self.assertEqual(mapper.label(0)["id"], 8)
        self.assertEqual(overrides, {"person": 8})
        self.assertIn('"person" = 8', output.getvalue())

    def test_noninteractive_ambiguity_and_user_cancel_stop(self):
        labels = LABELS + [{**LABELS[0], "id": 8, "name": "Person"}]
        with patch("main.sys.stdin.isatty", return_value=False):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                resolve_label_mapping({0: "person"}, labels, {}, Redactor())
        with patch("main.sys.stdin.isatty", return_value=True), patch("builtins.input", return_value=""):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                resolve_label_mapping({0: "person"}, labels, {}, Redactor())

    def test_explicit_skip_keeps_other_mapped_classes(self):
        labels = LABELS + [{"id": 9, "name": "Car", "type": "rectangle"}]
        mapper = LabelMapper({0: "person", 1: "car"}, labels, {"person": ""})
        self.assertIsNone(mapper.label(0))
        self.assertEqual(mapper.label(1)["id"], 9)

    def test_toml_accepts_label_ids_and_rejects_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.toml"
            path.write_text('[labels]\nperson = 8\n', encoding="utf-8")
            self.assertEqual(load_detection_config(path)[1], {"person": 8})
            path.write_text('[labels]\nperson = true\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_detection_config(path)

    def test_alias_case_and_shape_type(self):
        mapper = LabelMapper({0: "person", 1: "car", 2: "bicycle"}, LABELS + [
            {"id": 9, "name": "CAR", "type": "polygon"}])
        self.assertEqual(mapper.label(0)["id"], 7)
        self.assertIsNone(mapper.label(1))

    def test_ambiguous_alias_requires_override(self):
        labels = LABELS + [{**LABELS[0], "id": 8, "name": "person"}]
        with self.assertRaises(ValueError):
            LabelMapper({0: "person"}, labels)
        self.assertEqual(LabelMapper({0: "person"}, labels, {"person": "Pedestrian"}).label(0)["id"], 7)

    def test_unsupported_lane_and_wrong_override_stop(self):
        with self.assertRaises(ValueError):
            LabelMapper({0: "person"}, [{"id": 1, "name": "lane", "type": "polygon"}])
        with self.assertRaises(ValueError):
            LabelMapper({0: "person"}, LABELS, {"person": "unknown"})

    def test_attributes_require_real_defaults(self):
        label = {**LABELS[0], "attributes": [{"id": 4, "default_value": "false"}]}
        self.assertEqual(LabelMapper.attributes(label), [{"spec_id": 4, "value": "false"}])
        with self.assertRaises(ValueError):
            LabelMapper.attributes({"attributes": [{"id": 4}]})


class AnnotationTests(unittest.TestCase):
    def test_shape_clips_and_validates_pixel_coordinates(self):
        mapper = LabelMapper({0: "person"}, LABELS)
        shape = make_shape(10, Detection(0, .9, (-2., -4., 200., 90.)), LABELS[0], mapper, make_scope())
        self.assertEqual(shape["points"], [0., 0., 100., 80.])
        for points in ((1., 2., 1., 3.), (float("nan"), 1., 3., 4.)):
            with self.assertRaises(ValueError):
                make_shape(10, Detection(0, .9, points), LABELS[0], mapper, make_scope())

    def test_shape_rejects_outside_frames_labels_and_existing_ids(self):
        for shape in ({**box(), "frame": 0}, {**box(), "label_id": 999}, {**box(), "id": 42}):
            with self.assertRaises(ValueError):
                make_scope().validate_shape(shape)

    def test_deduplicates_current_run_and_existing_annotations(self):
        shape = box()
        self.assertEqual(len(deduplicate([shape, shape], {}, .85)[0]), 1)
        self.assertEqual(deduplicate([shape], {"shapes": [shape]}, .85), ([], 1))
        self.assertEqual(len(deduplicate([shape], {"shapes": [{**shape, "label_id": 8}]}, .85)[0]), 1)

    def test_track_interpolation_and_outside(self):
        track = {"label_id": 7, "shapes": [
            {"frame": 10, "type": "rectangle", "points": [0, 0, 10, 10], "outside": False},
            {"frame": 20, "type": "rectangle", "points": [10, 10, 20, 20], "outside": False},
            {"frame": 21, "type": "rectangle", "points": [10, 10, 20, 20], "outside": True},
        ]}
        self.assertEqual(list(existing_boxes({"tracks": [track]}, 15)), [(7, [5., 5., 15., 15.])])
        self.assertEqual(list(existing_boxes({"tracks": [track]}, 22)), [])

    def test_schema_requires_official_create_action(self):
        self.assertTrue(verify_create_schema(valid_schema()))
        schema = valid_schema()
        schema["paths"]["/api/jobs/{id}/annotations/"]["patch"]["parameters"][0]["schema"]["enum"] = ["update"]
        with self.assertRaises(ValueError):
            verify_create_schema(schema)

    def test_preservation_detects_modified_or_removed_old_objects(self):
        original = {"shapes": [{**box(), "id": 1}]}
        snapshot = annotation_snapshot(original)
        verify_preserved(snapshot, {"shapes": original["shapes"] + [{**box(11), "id": 2}]})
        with self.assertRaises(ValueError):
            verify_preserved(snapshot, {"shapes": []})
        with self.assertRaises(ValueError):
            verify_preserved(snapshot, {"shapes": [{**original["shapes"][0], "points": [1, 2, 3, 4]}]})

    def test_sdk_write_uses_only_patch_create_on_selected_job(self):
        client = CvatClient(Config(organization="cohort-5a"), Redactor())
        self.addCleanup(client.close)
        request = Mock(return_value=SimpleNamespace(status=200, data=b'{"shapes":[]}', headers={}, release_conn=Mock()))
        client.sdk.rest_client.pool_manager.request = request
        with contextlib.redirect_stdout(io.StringIO()):
            client.create_shapes([box()], scope=make_scope(), schema_verified=True, confirmed=True)
        call = request.call_args
        self.assertEqual(call.args[0], "PATCH")
        self.assertEqual(call.args[1], "https://cvat.edufun.ai/api/jobs/206/annotations/?action=create")
        body = json.loads(call.kwargs["body"])
        self.assertEqual(body["tracks"], [])
        self.assertEqual(body["tags"], [])
        self.assertNotIn("id", body["shapes"][0])

    def test_write_without_confirmation_or_with_empty_payload_stops(self):
        client = CvatClient(Config(backend="rest"), Redactor())
        self.addCleanup(client.close)
        client.session.request = Mock()
        for shapes, confirmed in (([box()], False), ([], True)):
            with self.assertRaises(ValueError):
                client.create_shapes(shapes, scope=make_scope(), schema_verified=True, confirmed=confirmed)
        client.session.request.assert_not_called()

    def test_apply_refreshes_and_verifies_preservation(self):
        client = Mock(config=Config(organization="cohort-5a"))
        client.job.return_value, client.organization.return_value = JOB, ORG
        client.metadata.return_value, client.job_labels.return_value = META, LABELS
        old = {**box(11), "id": 8}
        saved = {**box(), "id": 9}
        client.annotations.side_effect = [{"shapes": [old]}, {"shapes": [old]}, {"shapes": [old, saved]}]
        client.create_shapes.return_value = {"shapes": [saved]}
        with contextlib.redirect_stdout(io.StringIO()):
            created, _ = apply_shapes(client, [box()], make_scope(), DetectionConfig(), confirmed=True, schema_verified=True)
        self.assertEqual(created, 1)
        client.create_shapes.assert_called_once()

    def test_changed_metadata_blocks_writes(self):
        client = Mock(config=Config(organization="cohort-5a"))
        client.job.return_value, client.organization.return_value = JOB, ORG
        client.metadata.return_value = {**META, "deleted_frames": [10]}
        with self.assertRaises(ValueError):
            apply_shapes(client, [box()], make_scope(), DetectionConfig(), confirmed=True, schema_verified=True)
        client.create_shapes.assert_not_called()


class PipelineTests(unittest.TestCase):
    def test_all_flag_is_explicit_and_excludes_limit(self):
        _, args = parse_config(["--all"], annotation=True)
        self.assertTrue(args.all)
        self.assertFalse(args.apply)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_config(["--all", "--limit", "1"], annotation=True)

    def test_process_all_frames_visits_all_twenty_job_frames(self):
        scope = JobScope(206, {n: FrameInfo(n, 100, 80) for n in range(20)}, {7: LABELS[0]})
        raw = io.BytesIO()
        Image.new("RGB", (100, 80)).save(raw, format="PNG")
        client = Mock(redactor=Redactor())
        client.frame_bytes.return_value = raw.getvalue()
        detector = Mock(names={0: "person"})
        detector.predict.return_value = [[Detection(0, .9, (10., 10., 30., 40.))]]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            shapes, counts = process_frames(client, scope, detector, LabelMapper({0: "person"}, LABELS), DetectionConfig(), None)
        self.assertEqual(counts["processed"], 20)
        self.assertEqual([call.args[0] for call in client.frame_bytes.call_args_list], list(range(20)))
        self.assertEqual([shape["frame"] for shape in shapes], list(range(20)))

    def test_guided_cancel_stops_all_later_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            state = RunState({"job": 206}, Path(directory) / "state.json")
            for limit, apply, count in ((1, False, 1), (5, False, 5), (1, True, 1)):
                state.record(limit, apply, count)
            session = AnnotationSession(Mock(), Config(), make_scope(), Mock(), Mock(), DetectionConfig(), state, JOB)
            session.run_stage = Mock(return_value=StageResult(code=2))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(session.run_all(True), 2)
            session.run_stage.assert_called_once_with(5, True)
            self.assertNotIn("apply_5", state.completed)
            self.assertNotIn("apply_all", state.completed)

    def test_all_dry_mode_schedules_no_apply_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            state = RunState({"job": 206}, Path(directory) / "state.json")
            session = AnnotationSession(Mock(), Config(), make_scope(), Mock(), Mock(), DetectionConfig(), state, JOB)
            def dry_stage(limit, apply):
                self.assertFalse(apply)
                state.record(limit, False, min(limit or 20, 20))
                return StageResult(completed=True)
            session.run_stage = Mock(side_effect=dry_stage)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(session.run_all(False), 0)
            self.assertEqual([call.args for call in session.run_stage.call_args_list], [(1, False), (5, False), (None, False)])
            self.assertFalse(any(key.startswith("apply_") for key in state.completed))

    def test_duplicate_only_later_stage_is_verified_without_empty_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            state = RunState({"job": 206}, Path(directory) / "state.json")
            state.completed.update(dry_1=1, dry_5=2, apply_1=1)
            client = Mock()
            client.schema.return_value = valid_schema()
            existing = {"shapes": [{**box(), "id": 8}, {**box(11), "id": 9}]}
            session = AnnotationSession(client, Config(), make_scope(), Mock(), Mock(), DetectionConfig(), state, JOB, existing)
            counts = {"processed": 2, "detected": 2, "mapped": 2, "unmapped": 0, "invalid": 0}
            with patch("main.process_frames", return_value=([box(), box(11)], counts)):
                with patch("main.apply_shapes", return_value=(0, 0)) as writer:
                    with patch("main.sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="APPLY JOB 206"):
                        with contextlib.redirect_stdout(io.StringIO()):
                            self.assertTrue(session.run_stage(5, True).completed)
                    self.assertEqual(writer.call_args.args[1], [])
            self.assertEqual(state.completed["apply_5"], 2)
            client.create_shapes.assert_not_called()

    @patch("main.read_credentials")
    @patch("main.YoloDetector")
    @patch("main.CvatClient")
    def test_all_apply_from_one_frame_checkpoint_covers_twenty_with_one_login(self, client_type, detector_type, credentials):
        client = client_type.return_value
        client.redactor = Redactor()
        client.about.return_value = {"version": "2.76.0"}
        client.current_user.return_value = {"id": 212}
        client.job.return_value = {**JOB, "start_frame": 0, "stop_frame": 19}
        client.organization.return_value = ORG
        client.job_labels.return_value = LABELS
        client.metadata.return_value = {"start_frame": 0, "stop_frame": 19, "frame_filter": "", "size": 20,
                                        "deleted_frames": [], "included_frames": None,
                                        "frames": [{"name": "fake.png", "width": 100, "height": 80}] * 20}
        client.schema.return_value = valid_schema()
        detector_type.return_value.names = {0: "person"}
        detector_type.return_value.weights_hash = "fake-hash"
        first = {**box(), "frame": 0, "id": 100}
        stored = [first]
        client.annotations.side_effect = lambda: {"shapes": copy.deepcopy(stored)}
        stages, writes = [], []
        def fake_process(client, scope, detector, mapper, settings, limit):
            stages.append(limit)
            numbers = list(scope.frames)[:limit] if limit is not None else list(scope.frames)
            candidates = [{**box(), "frame": number} for number in numbers]
            return candidates, dict(processed=len(numbers), detected=len(numbers), mapped=len(numbers), unmapped=0, invalid=0)
        def fake_writer(client, candidates, scope, settings, **flags):
            self.assertTrue(flags["confirmed"])
            self.assertTrue(flags["schema_verified"])
            writes.append([shape["frame"] for shape in candidates])
            for shape in candidates:
                stored.append({**shape, "id": 100 + len(stored)})
            return len(candidates), 0
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            states = []
            def load_state(context):
                state = RunState(context, state_path)
                for limit, apply, count in ((1, False, 1), (5, False, 5), (1, True, 1)):
                    state.record(limit, apply, count)
                states.append(state)
                return state
            output = io.StringIO()
            with patch("main.RunState", side_effect=load_state), patch("main.process_frames", side_effect=fake_process):
                with patch("main.apply_shapes", side_effect=fake_writer):
                    with patch("main.sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="APPLY JOB 206") as confirm:
                        with contextlib.redirect_stdout(output):
                            self.assertEqual(main(["--all", "--apply", "--auth", "password"]), 0)
                        self.assertEqual(confirm.call_count, 2)
            self.assertEqual(states[0].completed["apply_all"], 20)
        self.assertEqual(stages, [5, None, None])
        self.assertEqual(writes, [list(range(1, 5)), list(range(5, 20))])
        self.assertEqual(sorted(shape["frame"] for shape in stored), list(range(20)))
        self.assertEqual(stored[0], first)
        client.login.assert_called_once()
        detector_type.assert_called_once()
        self.assertIn("toàn Job: 20 frame", output.getvalue())

    def test_completed_dry_runs_recommend_apply_one_then_follow_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            state = RunState({"job": 206}, Path(directory) / "state.json")
            self.assertEqual(state.next_action(), (1, False))
            state.record(1, False, 1)
            self.assertEqual(state.next_action(), (5, False))
            state.record(5, False, 5)
            self.assertEqual(state.next_action(), (1, True))
            state.require(*state.next_action())
            state.record(1, True, 1)
            self.assertEqual(state.next_action(), (5, True))
            state.record(5, True, 5)
            self.assertEqual(state.next_action(), (None, False))
            state.record(None, False, 10)
            self.assertEqual(state.next_action(), (None, True))
            state.record(None, True, 10)
            self.assertIsNone(state.next_action())

    def test_next_command_keeps_transport_scope_and_confirmation(self):
        state = Mock()
        state.next_action.return_value = (1, True)
        args = SimpleNamespace(auth="password", config=str(Path(__file__).parents[1] / "annotation.toml"))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_next_action(state, Config(backend="rest", frame_timeout=180), args)
        text = output.getvalue()
        for expected in ("--apply", "--limit '1'", "--organization 'cohort-5a'", "--backend 'rest'",
                         "--frame-timeout '180'", "APPLY JOB 206"):
            self.assertIn(expected, text)
        self.assertNotIn("--password", text)

    @patch("main.read_credentials")
    @patch("main.RunState")
    @patch("main.YoloDetector")
    @patch("main.CvatClient")
    def test_apply_confirmation_writes_via_verified_writer_and_records_success(self, client_type, detector_type, state_type, credentials):
        client = client_type.return_value
        client.redactor = Redactor()
        client.about.return_value = {"version": "2.76.0"}
        client.current_user.return_value = {"id": 212}
        client.job.return_value, client.organization.return_value = JOB, ORG
        client.job_labels.return_value, client.metadata.return_value = LABELS, META
        client.annotations.return_value = {"shapes": []}
        client.schema.return_value = valid_schema()
        detector_type.return_value.names = {0: "person"}
        detector_type.return_value.weights_hash = "fake-hash"
        state_type.return_value.next_action.return_value = (5, True)
        counts = {"processed": 1, "detected": 1, "mapped": 1, "unmapped": 0, "invalid": 0}
        output = io.StringIO()
        with patch("main.process_frames", return_value=([box()], counts)):
            with patch("main.apply_shapes", return_value=(1, 0)) as writer:
                with patch("main.sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="APPLY JOB 206"):
                    with contextlib.redirect_stdout(output):
                        self.assertEqual(main(["--limit", "1", "--apply"]), 0)
                writer.assert_called_once()
                self.assertIs(writer.call_args.kwargs["confirmed"], True)
                self.assertIs(writer.call_args.kwargs["schema_verified"], True)
        state_type.return_value.record.assert_called_once_with(1, True, 1)
        self.assertIn("https://cvat.edufun.ai/tasks/71/jobs/206", output.getvalue())
        self.assertIn("Đã ghi và đọc lại xác minh 1 rectangle", output.getvalue())

    @patch("main.read_credentials")
    @patch("main.RunState")
    @patch("main.YoloDetector")
    @patch("main.CvatClient")
    def test_apply_cancel_never_calls_writer(self, client_type, detector_type, state_type, credentials):
        client = client_type.return_value
        client.redactor = Redactor()
        client.about.return_value = {"version": "2.76.0"}
        client.current_user.return_value = {"id": 212}
        client.job.return_value, client.organization.return_value = JOB, ORG
        client.job_labels.return_value, client.metadata.return_value = LABELS, META
        client.annotations.return_value = {"shapes": []}
        client.schema.return_value = valid_schema()
        detector_type.return_value.names = {0: "person"}
        detector_type.return_value.weights_hash = "fake-hash"
        counts = {"processed": 1, "detected": 1, "mapped": 1, "unmapped": 0, "invalid": 0}
        with patch("main.process_frames", return_value=([box()], counts)), patch("main.apply_shapes") as writer:
            with patch("main.sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="cancel"):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["--limit", "1", "--apply"]), 2)
            writer.assert_not_called()
        state_type.return_value.record.assert_not_called()
        client.create_shapes.assert_not_called()

    def test_config_rejects_nonfinite_and_boolean_thresholds(self):
        for value in (float("nan"), float("inf"), True, 0, 1.1):
            with self.assertRaises(ValueError):
                DetectionConfig(confidence=value)

    def test_state_enforces_staged_dry_run_and_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            state = RunState({"job": 206}, Path(directory) / "state.json")
            state.require(1, False)
            with self.assertRaises(ValueError):
                state.require(5, False)
            state.record(1, False, 1)
            state.require(5, False)
            state.record(5, False, 2)
            with self.assertRaises(ValueError):
                state.require(5, True)
            state.record(1, True, 1)
            state.require(5, True)
            changed = RunState({"job": 207}, state.path)
            with self.assertRaises(ValueError):
                changed.require(5, False)

    def test_pipeline_uses_memory_and_only_selected_frame(self):
        raw = io.BytesIO()
        Image.new("RGB", (100, 80)).save(raw, format="PNG")
        client = Mock(redactor=Redactor())
        client.frame_bytes.return_value = raw.getvalue()
        detector = Mock(names={0: "person"})
        detector.predict.return_value = [[Detection(0, .9, (10., 10., 30., 40.))]]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            shapes, counts = process_frames(client, make_scope(), detector, LabelMapper({0: "person"}, LABELS), DetectionConfig(), 1)
        self.assertEqual(counts["processed"], 1)
        self.assertEqual(shapes[0]["frame"], 10)
        client.frame_bytes.assert_called_once_with(10, make_scope().frames)
        client.create_shapes.assert_not_called()

    @patch("main.read_credentials")
    @patch("main.YoloDetector")
    @patch("main.CvatClient")
    def test_main_dry_run_job_only_never_writes(self, client_type, detector_type, credentials):
        client = client_type.return_value
        client.redactor = Redactor()
        client.about.return_value = {"version": "2.76.0"}
        client.current_user.return_value = {"id": 212}
        client.job.return_value, client.organization.return_value = JOB, ORG
        client.job_labels.return_value, client.metadata.return_value = LABELS, META
        client.annotations.return_value = {"shapes": [], "tracks": []}
        raw = io.BytesIO()
        Image.new("RGB", (100, 80)).save(raw, format="PNG")
        client.frame_bytes.return_value = raw.getvalue()
        detector = detector_type.return_value
        detector.names, detector.weights_hash = {0: "person"}, "fake-weights-hash"
        detector.predict.return_value = [[Detection(0, .9, (10., 10., 30., 40.))]]
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            with patch("main.RunState", side_effect=lambda context: RunState(context, state_path)):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["--organization", "cohort-5a", "--limit", "1"]), 0)
            self.assertNotIn("fake.png", state_path.read_text())
        client.task.assert_not_called()
        client.create_shapes.assert_not_called()
        client.schema.assert_not_called()

    @patch("main.read_credentials")
    @patch("main.YoloDetector")
    @patch("main.CvatClient")
    def test_failed_auth_never_loads_model(self, client_type, detector_type, credentials):
        from auth import DiagnosticError
        client_type.return_value.about.return_value = {"version": "2.76.0"}
        client_type.return_value.login.side_effect = DiagnosticError("Đăng nhập", 401, "Invalid credentials")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--limit", "1"]), 1)
        detector_type.assert_not_called()


if __name__ == "__main__":
    unittest.main()
