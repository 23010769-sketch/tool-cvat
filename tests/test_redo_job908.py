"""Replacement integrity: real RLE, concurrent changes, interrupted saves, no network."""

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from cvat_sdk.masks import encode_mask

from config import Config
from auth import Credentials
import redo_job908 as redo


def schema():
    properties = {key: {} for key in ("id", "frame", "label_id", "points")}
    properties.update(type={"enum": ["rectangle", "mask"]}, source={"type": "string"})
    operation = {"operationId": "jobs_partial_update_annotations",
                 "parameters": [{"name": "action", "in": "query", "required": True,
                                 "schema": {"enum": ["create", "update", "delete"]}}],
                 "requestBody": {"content": {"application/json": {"schema": {"properties": {
                     "shapes": {"type": "array", "items": {"properties": properties}}}}}}}}
    return {"info": {"version": "2.76.0"}, "paths": {"/api/jobs/{id}/annotations/": {"patch": operation}}}


class RedoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input_patch = patch.object(redo, "INPUT", self.root)
        self.output_patch = patch.object(redo, "OUTPUT", self.root)
        self.input_patch.start(); self.output_patch.start()
        self.addCleanup(self.input_patch.stop); self.addCleanup(self.output_patch.stop)
        self.labels = [{"id": 100+k, "name": name, "type": "any", "attributes": []}
                       for k, name in enumerate(redo.NAMES)]
        (self.root / "labels.json").write_text(json.dumps(self.labels), encoding="utf-8")
        self.job = {"id": 908, "task_id": 211, "organization": 1, "type": "annotation", "stage": "annotation",
                    "dimension": "2d", "media_type": "image", "mode": "annotation", "start_frame": 80,
                    "stop_frame": 80, "assignee": {"id": 212}}
        self.meta = {"start_frame": 80, "stop_frame": 80, "size": 1, "frame_filter": "", "deleted_frames": [],
                     "frames": [{"name": "fake.png", "width": 8, "height": 6}]}
        self.scope = redo.make_scope(self.job, self.meta, self.labels)
        old = {"id": 22, "frame": 80, "label_id": 113, "type": "rectangle", "source": "auto",
               "points": [1., 1., 7., 5.], "occluded": False, "rotation": 0., "attributes": [], "z_order": 0}
        self.original = {"version": 0, "shapes": [old], "tags": [], "tracks": []}
        (self.root / "annotations_before.json").write_text(json.dumps(self.original), encoding="utf-8")
        self.shapes = []
        for class_id, bitmap in ((0, np.indices((6,8))[1] < 4), (13, np.indices((6,8))[1] >= 4)):
            self.shapes.append({"frame": 80, "label_id": 100+class_id, "type": "mask", "source": "auto",
                                "points": encode_mask(bitmap), "occluded": False, "rotation": 0.,
                                "z_order": 0, "attributes": []})
        buffer = io.BytesIO()
        with Image.new("RGB", (8,6), (30,60,90)) as image:
            image.save(buffer, format="PNG")
            image_hash = redo.pixel_digest(image)
        self.raw = buffer.getvalue()
        self.plan = {"plan_id": "test-reviewed-plan", "original_digest": redo.annotation_digest(self.original),
                     "original_ids": [22], "labels_digest": redo.digest(self.labels), "shapes": self.shapes,
                     "frames": [{"frame": 80, "width": 8, "height": 6, "pixels_sha256": image_hash}],
                     "review_notes": []}
        self.current = copy.deepcopy(self.original)
        self.client = Mock(config=Config(task_id=211, job_id=908, backend="rest"))
        self.client.job.return_value = self.job
        self.client.organization.return_value = {"id": 1, "slug": "cohort-5a"}
        self.client.metadata.return_value = self.meta
        self.client.job_labels.return_value = self.labels
        self.client.schema.return_value = schema()
        self.client.frame_bytes.return_value = self.raw
        self.client.annotations.side_effect = lambda: copy.deepcopy(self.current)

        def create(shapes, **kwargs):
            self.current["shapes"].extend(dict(s, id=500+k) for k,s in enumerate(copy.deepcopy(shapes)))
            return {"shapes": self.current["shapes"][1:]}

        def delete(step, call, method, path, **kwargs):
            self.assertEqual((method, path, kwargs["params"]), ("PATCH", "/api/jobs/908/annotations/", {"action":"delete"}))
            self.assertEqual({s["id"] for s in kwargs["json"]["shapes"]}, {22})
            self.current["shapes"] = [s for s in self.current["shapes"] if s["id"] != 22]
            return {"shapes": []}

        self.client.create_shapes.side_effect = create
        self.client._call.side_effect = delete

    def apply(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return redo.apply_plan(self.client, {"id":212}, self.plan)

    def test_creates_verifies_then_deletes_only_original_ids(self):
        fresh = self.apply()
        self.assertEqual(len(fresh), 2)
        self.client.create_shapes.assert_called_once()
        self.client._call.assert_called_once()
        self.assertEqual({s["id"] for s in self.current["shapes"]}, {500,501})
        self.assertTrue((self.root / "before_apply_test-reviewe.json").exists())

    def test_does_not_delete_when_creation_is_partial(self):
        self.client.create_shapes.side_effect = lambda shapes, **kw: self.current["shapes"].append(dict(shapes[0],id=500))
        with self.assertRaisesRegex(ValueError, "ghi chưa đủ"):
            self.apply()
        self.client._call.assert_not_called()
        self.assertIn(22, [s["id"] for s in self.current["shapes"]])

    def test_resume_after_create_does_not_create_duplicates(self):
        self.current["shapes"].extend(dict(s,id=500+k) for k,s in enumerate(self.shapes))
        self.apply()
        self.client.create_shapes.assert_not_called()
        self.client._call.assert_called_once()

    def test_rerun_after_success_does_not_write_annotations(self):
        self.current["shapes"] = [dict(s,id=500+k) for k,s in enumerate(self.shapes)]
        self.apply()
        self.client.create_shapes.assert_not_called()
        self.client._call.assert_not_called()

    def test_changed_pixels_block_all_writes(self):
        changed = io.BytesIO()
        Image.new("RGB", (8,6), "red").save(changed,format="PNG")
        self.client.frame_bytes.return_value = changed.getvalue()
        with self.assertRaisesRegex(ValueError, "Ảnh trên CVAT khác"):
            self.apply()
        self.client.create_shapes.assert_not_called()
        self.client._call.assert_not_called()

    def test_concurrent_changes_or_extra_objects_stop_writes(self):
        for kind in ("edit", "extra", "track"):
            self.current = copy.deepcopy(self.original)
            if kind == "edit": self.current["shapes"][0]["points"][0] = 2.
            if kind == "extra": self.current["shapes"].append(dict(self.original["shapes"][0],id=23))
            if kind == "track": self.current["tracks"] = [{"id":1000}]
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.apply()
        self.client.create_shapes.assert_not_called()
        self.client._call.assert_not_called()

    def test_changed_assignment_blocks_writes(self):
        self.client.job.return_value = {**self.job, "assignee":{"id":999}}
        with self.assertRaisesRegex(ValueError, "không được giao"):
            self.apply()
        self.client.create_shapes.assert_not_called()
        self.client._call.assert_not_called()

    def test_schema_must_explicitly_support_delete(self):
        changed = schema()
        changed["paths"]["/api/jobs/{id}/annotations/"]["patch"]["parameters"][0]["schema"]["enum"].remove("delete")
        self.client.schema.return_value = changed
        with self.assertRaisesRegex(ValueError, "không xác nhận xóa"):
            self.apply()
        self.client.create_shapes.assert_not_called()

    def test_missing_semantic_label_or_wrong_geometry_is_rejected(self):
        for labels in (self.labels[:-1], [dict(s,type="rectangle") for s in self.labels]):
            with self.assertRaises(ValueError): redo.semantic_labels(labels)

    def test_hand_drawn_originals_are_not_replaced(self):
        original = copy.deepcopy(self.original)
        original["shapes"][0]["source"] = "manual"
        with self.assertRaisesRegex(ValueError, "nhãn vẽ tay"):
            redo.validate_originals(original, self.scope.frames)

    def test_cli_default_preview_never_prompts_or_connects(self):
        with patch.object(redo,"load_plan",return_value=self.plan), patch.object(redo,"read_credentials") as auth, \
                patch.object(redo,"CvatClient") as client, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(redo.main([]),0)
        auth.assert_not_called()
        client.assert_not_called()

    def test_cli_apply_uses_terminal_credentials_and_fixed_job_scope(self):
        with patch.object(redo,"load_plan",return_value=self.plan), \
                patch.object(redo,"read_credentials",return_value=Credentials("password",username="fake-user",password="fake-secret")) as auth, \
                patch.object(redo,"CvatClient",return_value=self.client) as factory, \
                patch.object(redo,"apply_plan",return_value=[]), patch.object(redo,"post_review_issues",return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.client.current_user.return_value={"id":212}
            self.assertEqual(redo.main(["--apply"]),0)
        config=factory.call_args.args[0]
        self.assertEqual((config.task_id,config.job_id,config.organization,config.backend),(211,908,"cohort-5a","rest"))
        self.assertEqual(auth.call_args.args[:2],("password",None))
        self.assertNotIn("fake-secret",output.getvalue())
        self.client.close.assert_called_once()

    def test_cli_label_audit_does_not_need_plan_or_frames(self):
        with patch.object(redo,"load_plan") as plan, \
                patch.object(redo,"read_credentials",return_value=Credentials("password",username="fake-user",password="fake-secret")), \
                patch.object(redo,"CvatClient",return_value=self.client), contextlib.redirect_stdout(io.StringIO()):
            self.client.current_user.return_value={"id":212}
            self.assertEqual(redo.main(["--audit-labels"]),0)
        plan.assert_not_called()
        self.client.frame_bytes.assert_not_called()
        self.client.create_shapes.assert_not_called()
        report=json.loads((self.root/"semantic_label_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["labels"]),19)

    def test_second_apply_is_blocked_until_first_releases_lock(self):
        with redo.exclusive_apply():
            with self.assertRaisesRegex(ValueError,"đang chạy"):
                with redo.exclusive_apply():
                    self.fail("Concurrent replacement must not start")
        with redo.exclusive_apply():
            pass

    def test_label_order_colors_and_response_metadata_do_not_block_replacement(self):
        changed = [dict(label, color="#123456", url=f"/api/labels/{label['id']}", has_parent=False,
                        project_id=44, updated_date="display-metadata") for label in reversed(self.labels)]
        changed.append({"id":9000,"name":"lane/unused","type":"polyline","attributes":[]})
        self.client.job_labels.return_value = changed
        self.apply()
        self.client.create_shapes.assert_called_once()
        self.client._call.assert_called_once()
        report = redo.read_json(self.root / "label_comparison.json")
        self.assertTrue(report["compatible"])
        self.assertTrue(report["raw_metadata_changed"])
        self.assertEqual(report["differences"], [])

    def test_restricting_any_geometry_to_mask_is_compatible(self):
        self.client.job_labels.return_value = [dict(label,type="mask") for label in self.labels]
        self.apply()
        self.client.create_shapes.assert_called_once()

    def test_real_label_id_change_reports_the_label_and_prevents_writes(self):
        changed = copy.deepcopy(self.labels)
        changed[0]["id"] = 9001
        self.client.job_labels.return_value = changed
        with self.assertRaisesRegex(ValueError,"road.*id"):
            self.apply()
        self.client.create_shapes.assert_not_called()
        self.client._call.assert_not_called()
        report = redo.read_json(self.root / "label_comparison.json")
        self.assertFalse(report["compatible"])
        self.assertEqual(report["differences"], [{"name":"road","fields":["id"]}])

    def test_added_checkbox_defaults_are_sent_on_masks(self):
        changed = copy.deepcopy(self.labels)
        changed[0]["attributes"] = [{"id":7,"name":"reviewed","input_type":"checkbox",
                                     "default_value":"false","values":["true","false"],"mutable":False}]
        self.client.job_labels.return_value = changed
        self.apply()
        self.client.create_shapes.assert_called_once()
        created = self.client.create_shapes.call_args.args[0]
        road = next(s for s in created if s['label_id']==100)
        self.assertEqual(road['attributes'],[{'spec_id':7,'value':'false'}])
        self.client._call.assert_called_once()

    def test_invalid_added_default_prevents_writes(self):
        changed = copy.deepcopy(self.labels)
        changed[0]['attributes']=[{'id':7,'name':'occluded','input_type':'checkbox',
                                   'default_value':'unknown','values':['false'],'mutable':True}]
        self.client.job_labels.return_value=changed
        with self.assertRaisesRegex(ValueError,'không hợp lệ'):
            self.apply()
        self.client.create_shapes.assert_not_called()
        self.client._call.assert_not_called()

    def test_materialized_defaults_on_originals_are_preserved_during_replacement(self):
        changed=copy.deepcopy(self.labels)
        changed[13]['attributes']=[{'id':7,'name':'occluded','input_type':'checkbox',
                                    'default_value':'false','values':['false'],'mutable':True},
                                   {'id':8,'name':'truncated','input_type':'checkbox',
                                    'default_value':'false','values':['false'],'mutable':True}]
        self.client.job_labels.return_value=changed
        self.current['shapes'][0]['attributes']=[{'spec_id':7,'value':'false'},{'spec_id':8,'value':'false'}]
        self.apply()
        self.client._call.assert_called_once()
        old=self.client._call.call_args.kwargs['json']['shapes'][0]
        self.assertEqual(len(old['attributes']),2)

    def test_nondefault_user_attribute_change_on_original_blocks_deletion(self):
        changed=copy.deepcopy(self.labels)
        changed[13]['attributes']=[{'id':7,'name':'occluded','input_type':'checkbox',
                                    'default_value':'false','values':['false'],'mutable':True}]
        self.client.job_labels.return_value=changed
        self.current['shapes'][0]['attributes']=[{'spec_id':7,'value':'true'}]
        with self.assertRaisesRegex(ValueError,'Box cũ đã bị sửa'):
            self.apply()
        self.client.create_shapes.assert_not_called()
        self.client._call.assert_not_called()

    def test_changes_to_reviewed_attribute_definition_are_still_blocked(self):
        before=copy.deepcopy(self.labels)
        before[0]['attributes']=[{'id':7,'name':'occluded','input_type':'checkbox',
                                  'default_value':'false','values':['false'],'mutable':True}]
        self.plan['semantic_label_contract']=redo.semantic_label_contract(before)
        after=copy.deepcopy(before)
        after[0]['attributes'][0]['default_value']='true'
        self.client.job_labels.return_value=after
        with self.assertRaisesRegex(ValueError,'road.*attributes'):
            self.apply()
        self.client.create_shapes.assert_not_called()

    def test_renamed_label_is_not_silently_remapped(self):
        changed = copy.deepcopy(self.labels)
        changed[0]["name"] = "not-road"
        self.client.job_labels.return_value = changed
        with self.assertRaisesRegex(ValueError,"road"):
            self.apply()
        self.client.create_shapes.assert_not_called()

    def test_attribute_and_value_order_are_not_schema_changes(self):
        attribute = {"id":7,"name":"kind","input_type":"select","mutable":False,
                     "default_value":"a","values":["a","b"]}
        first = copy.deepcopy(self.labels)
        first[0]["attributes"] = [attribute,dict(attribute,id=8,name="second")]
        second = copy.deepcopy(first)
        second[0]["attributes"].reverse()
        for item in second[0]["attributes"]:
            item["values"].reverse()
        self.assertEqual(redo.semantic_label_contract(first),redo.semantic_label_contract(second))

    def test_legacy_plan_baseline_must_still_match_its_recorded_digest(self):
        changed = copy.deepcopy(self.labels)
        changed[0]["id"] = 9001
        (self.root / "labels.json").write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"Bản sao labels khác"):
            self.apply()
        self.client.create_shapes.assert_not_called()

    def test_cleanup_deletes_only_own_generated_issues_for_this_job(self):
        self.plan['review_notes']=[{'frame':80,'type':'LOW_VISIBILITY','message':'Check boundary','position':[0,0,8,6]}]
        marker=f"[Job908 semantic {self.plan['plan_id'][:12]}] LOW_VISIBILITY: Check boundary"
        own={'id':837,'job':908,'frame':80,'owner':{'id':212}}
        manual={'id':900,'job':908,'frame':80,'owner':{'id':212}}
        foreign={'id':901,'job':908,'frame':80,'owner':{'id':999}}
        other_job={'id':902,'job':909,'frame':80,'owner':{'id':212}}
        other_frame={'id':903,'job':908,'frame':81,'owner':{'id':212}}
        issues=[own,manual,foreign,other_job,other_frame]
        comments=[{'issue':837,'message':marker},{'issue':900,'message':'Manual review'},
                  {'issue':901,'message':marker},{'issue':902,'message':marker},{'issue':903,'message':marker}]
        self.client.schema.return_value={'paths':{'/api/issues/{id}':{'delete':{'operationId':'issues_destroy'}}}}
        self.client._list.side_effect=lambda step,call,path,filters: copy.deepcopy(comments if path=='/api/comments' else issues)
        calls=[]
        def call(step,callback,method,path,**kwargs):
            calls.append((method,path))
            self.assertEqual(path,'/api/issues/837')
            if method=='GET': return copy.deepcopy(own)
            self.assertEqual(method,'DELETE')
            self.assertTrue(kwargs['binary'])
            issues.remove(own)
            return b''
        self.client._call.side_effect=call
        before=copy.deepcopy(self.current)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(redo.remove_auto_issues(self.client,{'id':212},self.plan),[837])
        self.assertEqual(calls,[('GET','/api/issues/837'),('DELETE','/api/issues/837')])
        self.assertEqual({i['id'] for i in issues},{900,901,902,903})
        self.assertEqual(self.current,before)
        self.client.create_shapes.assert_not_called()
        report=redo.read_json(self.root/'issue_cleanup_status.json')
        self.assertTrue(report['annotations_unchanged'])

    def test_cleanup_refuses_issue_that_changed_job_before_delete(self):
        self.plan['review_notes']=[{'frame':80,'type':'LOW_VISIBILITY','message':'Check','position':[0,0,8,6]}]
        marker=f"[Job908 semantic {self.plan['plan_id'][:12]}] LOW_VISIBILITY: Check"
        self.client.schema.return_value={'paths':{'/api/issues/{id}':{'delete':{'operationId':'issues_destroy'}}}}
        self.client._list.side_effect=[
            [{'id':837,'job':908,'frame':80,'owner':{'id':212}}],
            [{'issue':837,'message':marker}]]
        self.client._call.side_effect=None
        self.client._call.return_value={'id':837,'job':909,'frame':80,'owner':{'id':212}}
        with self.assertRaisesRegex(ValueError,'Issue thay đổi'):
            redo.remove_auto_issues(self.client,{'id':212},self.plan)
        self.assertEqual(self.client._call.call_count,1)
        self.assertEqual(self.client._call.call_args.args[2],'GET')

    def test_cleanup_cli_never_runs_annotation_apply_or_frame_download(self):
        with patch.object(redo,'load_plan',return_value=self.plan), \
                patch.object(redo,'read_credentials',return_value=Credentials('password',username='fake',password='fake')), \
                patch.object(redo,'CvatClient',return_value=self.client), \
                patch.object(redo,'apply_plan') as apply, \
                patch.object(redo,'remove_auto_issues',return_value=[837]) as remove, \
                patch.object(redo,'post_review_issues') as post,contextlib.redirect_stdout(io.StringIO()):
            self.client.current_user.return_value={'id':212}
            self.assertEqual(redo.main(['--remove-auto-issues']),0)
        remove.assert_called_once()
        apply.assert_not_called()
        post.assert_not_called()
        self.client.frame_bytes.assert_not_called()
        self.client.create_shapes.assert_not_called()
        self.assertFalse((self.root/'apply_status.json').exists())

    def test_normal_apply_no_longer_posts_review_issues(self):
        with patch.object(redo,'load_plan',return_value=self.plan), \
                patch.object(redo,'read_credentials',return_value=Credentials('password',username='fake',password='fake')), \
                patch.object(redo,'CvatClient',return_value=self.client), \
                patch.object(redo,'apply_plan'),patch.object(redo,'remove_auto_issues',return_value=[]), \
                patch.object(redo,'post_review_issues') as post,contextlib.redirect_stdout(io.StringIO()):
            self.client.current_user.return_value={'id':212}
            self.assertEqual(redo.main(['--apply']),0)
        post.assert_not_called()


if __name__=="__main__":
    unittest.main()
