"""Unit tests dùng dữ liệu giả; không gọi server của trường."""

import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auth import Credentials, DiagnosticError, Redactor, read_credentials
from check_api import main, validate_scope
from config import Config, parse_config
from permission_diagnostics import diagnose_permissions
from cvat_client import CvatClient


def response(status, data):
    result = Mock()
    result.status_code = status
    result.content = json.dumps(data).encode()
    return result


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.redactor = Redactor()
        self.client = CvatClient(Config(backend="rest"), self.redactor)
        self.client.session.request = Mock()
        self.addCleanup(self.client.close)

    def test_no_secret_in_errors(self):
        self.redactor.add("test-secret-value")
        text = self.redactor.body(json.dumps({
            "detail": "invalid test-secret-value", "nested": {"password": "unknown-secret"},
            "sessionid": "session-secret", "token": "token-secret",
        }))
        for secret in ("test-secret-value", "unknown-secret", "session-secret", "token-secret"):
            self.assertNotIn(secret, text)
        self.assertIn("invalid", text)

    def test_non_json_error_is_never_printed(self):
        self.assertNotIn("secret-data", self.redactor.body("<html>secret-data</html>"))

    def test_login_400_preserves_detail_and_stops(self):
        self.client.session.request.return_value = response(400, {"non_field_errors": ["Unable to log in."]})
        with self.assertRaises(DiagnosticError) as caught:
            self.client.login(Credentials("password", "fake-user", "fake-password"))
        self.assertEqual(caught.exception.status, 400)
        self.assertIn("Unable to log in", caught.exception.detail)
        self.assertEqual(self.client.session.request.call_count, 1)

    def test_403_is_not_reported_as_invalid_password(self):
        self.client.session.request.return_value = response(403, {"detail": "Permission denied."})
        with self.assertRaises(DiagnosticError) as caught:
            self.client.task()
        self.assertEqual(caught.exception.status, 403)
        self.assertEqual(caught.exception.step, "Task 71")

    def test_pat_is_bearer_and_requires_user_verification(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.client.login(Credentials("pat", token="fake-pat"))
        self.assertEqual(self.client.session.headers["Authorization"], "Bearer fake-pat")
        self.client.session.request.assert_not_called()

    def test_api_session_uses_only_api_cookies(self):
        def login_response(*args, **kwargs):
            self.client.session.cookies.set("sessionid", "fake-session")
            self.client.session.cookies.set("csrftoken", "fake-csrf")
            return response(200, {"key": "unused-legacy-token"})
        self.client.session.request.side_effect = login_response
        with contextlib.redirect_stdout(io.StringIO()):
            self.client.login(Credentials("session", "fake-user", "fake-password"))
        self.assertEqual(self.client.session.headers["X-CSRFToken"], "fake-csrf")
        self.assertEqual(self.client.session.headers["Origin"], "https://cvat.edufun.ai")
        self.assertNotIn("Authorization", self.client.session.headers)

    def test_tls_and_redirect_guards(self):
        self.client.session.request.return_value = response(302, {"detail": "redirect"})
        with self.assertRaises(DiagnosticError) as caught:
            self.client.about()
        self.assertEqual(caught.exception.status, 302)
        params = self.client.session.request.call_args.kwargs
        self.assertIs(params["verify"], True)
        self.assertIs(params["allow_redirects"], False)
        self.assertEqual(params["timeout"], 30)

    def test_labels_paginate_without_following_external_url(self):
        self.client.session.request.side_effect = [
            response(200, {"results": [{"id": 1, "name": "car"}], "next": "https://untrusted.invalid"}),
            response(200, {"results": [{"id": 2, "name": "person"}], "next": None}),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(len(self.client.labels()), 2)
        for call in self.client.session.request.call_args_list:
            self.assertEqual(call.args[1], "https://cvat.edufun.ai/api/labels")

    def test_rejects_wrong_task_org_and_job_type(self):
        task = {"id": 71, "organization": 4}
        job = {"id": 206, "task_id": 71, "type": "annotation"}
        org = {"id": 4, "slug": "cohort-5a"}
        config = Config(organization="cohort-5a")
        validate_scope(config, task, job, org)
        for bad_job, bad_org in [
            ({**job, "task_id": 72}, org),
            ({**job, "id": 207}, org),
            ({**job, "type": "ground_truth"}, org),
            (job, {**org, "slug": "another-org"}),
        ]:
            with self.assertRaises(DiagnosticError):
                validate_scope(config, task, bad_job, bad_org)

    def test_rejects_insecure_urls(self):
        for url in ("http://cvat.edufun.ai", "https://user:secret@cvat.edufun.ai",
                    "https://cvat.edufun.ai/api", "https://cvat.edufun.ai?token=secret"):
            with self.assertRaises(ValueError):
                Config(url=url)

    @patch.dict("os.environ", {}, clear=True)
    @patch("auth.sys.stdin")
    def test_noninteractive_credentials_stop_without_echo(self, stdin):
        stdin.isatty.return_value = False
        with self.assertRaises(ValueError):
            read_credentials("password", None, self.redactor)

    @patch("check_api.CvatClient")
    @patch("check_api.read_credentials")
    def test_failed_auth_never_proceeds_to_task(self, credentials, client_type):
        client = client_type.return_value
        client.about.return_value = {"version": "2.76.0"}
        client.login.side_effect = DiagnosticError("Đăng nhập", 400, "Unable to log in.")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([]), 1)
        client.task.assert_not_called()
        client.annotations.assert_not_called()
        client.metadata.assert_not_called()
        client.close.assert_called_once()

    @patch("check_api.CvatClient")
    def test_public_probe_never_logs_in(self, client_type):
        client = client_type.return_value
        client.about.return_value = {"version": "2.76.0"}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--public-only"]), 0)
        client.login.assert_not_called()
        client.task.assert_not_called()

    def test_sdk_calls_and_configuration_match_installed_version(self):
        sdk_client = CvatClient(Config(), Redactor())
        self.addCleanup(sdk_client.close)
        sdk_client.sdk.rest_client.pool_manager.request = Mock(return_value=SimpleNamespace(
            status=200, data=b'{"version":"2.76.0"}', headers={},
            release_conn=Mock(), getheaders=lambda: {},
        ))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sdk_client.about()["version"], "2.76.0")
        self.assertTrue(sdk_client.sdk.configuration.verify_ssl)
        self.assertEqual(sdk_client.sdk.configuration.retries.redirect, 0)
        self.assertEqual(sdk_client.sdk.configuration.retries.total, 0)

    def test_sdk_session_and_all_read_endpoints(self):
        from urllib3._collections import HTTPHeaderDict
        sdk_client = CvatClient(Config(organization="cohort-5a"), self.redactor)
        self.addCleanup(sdk_client.close)
        pool_request = Mock()
        sdk_client.sdk.rest_client.pool_manager.request = pool_request
        def sdk_response(data, cookies=False):
            headers = HTTPHeaderDict()
            if cookies:
                headers.add("Set-Cookie", "sessionid=fake-session; Path=/; Secure")
                headers.add("Set-Cookie", "csrftoken=fake-csrf; Path=/; Secure")
            return SimpleNamespace(status=200, data=json.dumps(data).encode(),
                                   headers=headers, release_conn=Mock())
        pool_request.side_effect = [
            sdk_response({"key": "unused-token"}, cookies=True),
            sdk_response({"id": 8}), sdk_response({"id": 71}),
            sdk_response({"id": 4, "slug": "cohort-5a"}), sdk_response({"id": 206}),
            sdk_response({"results": [{"id": 1, "name": "car"}], "next": None}),
            sdk_response({"shapes": [], "tracks": []}), sdk_response({"frames": []}),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            sdk_client.login(Credentials("password", "fake-user", "fake-password"))
            self.assertEqual(sdk_client.current_user()["id"], 8)
            sdk_client.set_organization()
            sdk_client.task()
            sdk_client.organization(4)
            sdk_client.job()
            sdk_client.labels()
            sdk_client.annotations()
            sdk_client.metadata()
        calls = pool_request.call_args_list
        self.assertEqual(len(calls), 8)
        self.assertIn("sessionid=fake-session", calls[1].kwargs["headers"]["Cookie"])
        self.assertEqual(calls[2].kwargs["headers"]["X-Organization"], "cohort-5a")
        self.assertEqual(calls[1].kwargs["headers"]["X-CSRFToken"], "fake-csrf")
        self.assertEqual([call.args[0] for call in calls], ["POST"] + ["GET"] * 7)

    def test_personal_removes_org_header_in_both_backends(self):
        for backend in ("rest", "sdk"):
            with self.subTest(backend=backend):
                client = CvatClient(Config(backend=backend, organization=""), self.redactor)
                self.addCleanup(client.close)
                headers = client.sdk.default_headers if client.sdk else client.session.headers
                headers["X-Organization"] = "cohort-5a"
                client.set_organization()
                self.assertNotIn("X-Organization", headers)

    @patch.dict("os.environ", {"CVAT_ORGANIZATION": "cohort-5a"})
    def test_personal_flag_overrides_old_environment(self):
        config, _ = parse_config(["--personal"])
        self.assertEqual(config.organization, "")

    def test_personal_scope_rejects_organization_resources(self):
        task = {"id": 71, "organization": None}
        job = {"id": 206, "task_id": 71, "type": "annotation", "organization": None}
        validate_scope(Config(organization=""), task, job, {})
        with self.assertRaises(DiagnosticError):
            validate_scope(Config(organization=""), {**task, "organization": 4}, job, {})
        with self.assertRaises(DiagnosticError):
            validate_scope(Config(organization=""), task, {**job, "organization": 4}, {})

    @patch("check_api.CvatClient")
    @patch("check_api.read_credentials")
    def test_short_secret_does_not_hide_local_task_id(self, credentials, client_type):
        client = client_type.return_value
        client.about.return_value = {"version": "2.76.0"}
        client.current_user.return_value = {"id": 212, "username": "fake-user"}
        client.task.side_effect = DiagnosticError("Task 71", 403, "Permission denied.")
        client.job.side_effect = DiagnosticError("Job 206", 403, "Permission denied.")
        with patch("check_api.Redactor") as redactor_type:
            redactor = Redactor()
            redactor.add("1")
            redactor_type.return_value = redactor
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["--personal"]), 1)
        self.assertIn("Bước: Task 71 | HTTP 403", output.getvalue())
        client.job.assert_called_once()
        client.metadata.assert_not_called()

    def test_task_403_can_probe_only_the_authorized_job(self):
        client = Mock()
        client.job.return_value = {"id": 206, "task_id": 71, "organization": None,
                                   "type": "annotation", "assignee": {"id": 212}}
        client.job_labels.return_value = []
        client.annotations.return_value = {"shapes": []}
        client.metadata.return_value = {"frames": []}
        with contextlib.redirect_stdout(io.StringIO()):
            result = diagnose_permissions(client, Config(organization=""), {"id": 212}, task_denied=True)
        self.assertEqual(result, 1)  # Task read is still unverified/denied.
        client.task.assert_not_called()
        client.organizations.assert_not_called()
        client.memberships.assert_not_called()
        client.job_labels.assert_called_once()

    def test_wrong_job_workspace_stops_dependent_reads(self):
        client = Mock()
        client.job.return_value = {"id": 206, "task_id": 71, "organization": 4, "type": "annotation"}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(DiagnosticError):
            diagnose_permissions(client, Config(organization=""), {"id": 212}, task_denied=True)
        client.job_labels.assert_not_called()
        client.annotations.assert_not_called()
        client.metadata.assert_not_called()

    def test_diagnostic_authentication_loss_stops_probes(self):
        client = Mock()
        client.organizations.side_effect = DiagnosticError("Organization", 401, "Expired session")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(DiagnosticError):
            diagnose_permissions(client, Config(organization="cohort-5a"), {"id": 212}, task_denied=True)
        client.job.assert_not_called()


if __name__ == "__main__":
    unittest.main()
