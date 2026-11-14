"""Lỗi mạng lồng nhau, timeout ảnh và REST fallback dùng cùng phiên API."""

import contextlib
import io
import json
import socket
import ssl
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from urllib3 import exceptions as ue

from auth import DiagnosticError, Redactor
from config import Config, parse_config
from cvat_client import CvatClient
from network_diagnostics import describe_transport_error


def read_error():
    return ue.MaxRetryError(None, "/private?token=never-print-this",
                            ue.ReadTimeoutError(None, "/private", "password=do-not-log"))


def raw_response(status, data):
    return SimpleNamespace(status=status, data=data, headers={}, release_conn=Mock())


class NetworkTests(unittest.TestCase):
    def sdk_client(self):
        client = CvatClient(Config(), Redactor())
        client.set_organization()
        client.sdk.cookies.load("sessionid=fake-api-session; csrftoken=fake-api-csrf")
        client.redactor.add("fake-api-session", "fake-api-csrf")
        self.addCleanup(client.close)
        return client

    def fallback_session(self):
        session = requests.Session()
        self.addCleanup(session.close)
        response = Mock(status_code=200, content=b"fake-image-bytes")
        session.request = Mock(return_value=response)
        return session

    def test_nested_read_timeout_is_classified_without_private_messages(self):
        code, detail = describe_transport_error(read_error(), (15, 120))
        self.assertEqual(code, "READ_TIMEOUT")
        self.assertIn("120s", detail)
        self.assertIn("MaxRetryError", detail)
        self.assertIn("ReadTimeoutError", detail)
        for private in ("never-print-this", "do-not-log", "/private"):
            self.assertNotIn(private, detail)

    def test_requests_wrapped_read_timeout_is_classified(self):
        wrapped = requests.exceptions.ConnectionError(read_error())
        self.assertEqual(describe_transport_error(wrapped, (15, 120))[0], "READ_TIMEOUT")

    def test_tls_and_dns_are_distinguished(self):
        cert = ue.MaxRetryError(None, "/", ue.SSLError(ssl.SSLCertVerificationError(1, "private message")))
        self.assertEqual(describe_transport_error(cert, 30)[0], "TLS_CERTIFICATE")
        dns = ue.MaxRetryError(None, "/", ue.NameResolutionError("host.invalid", None, socket.gaierror(-2, "private")))
        self.assertEqual(describe_transport_error(dns, 30)[0], "DNS")

    def test_get_frame_fallback_reuses_api_session_scope_and_timeout(self):
        client, fallback = self.sdk_client(), self.fallback_session()
        pool = Mock(side_effect=read_error())
        client.sdk.rest_client.pool_manager.request = pool
        with patch("cvat_client.requests.Session", return_value=fallback) as session_type:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(client.frame_bytes(10, {10, 11}), b"fake-image-bytes")
                self.assertEqual(client.frame_bytes(11, {10, 11}), b"fake-image-bytes")
        self.assertEqual(pool.call_count, 1)
        session_type.assert_called_once()
        self.assertEqual(fallback.request.call_count, 2)
        self.assertEqual(fallback.headers["X-Organization"], "cohort-5a")
        self.assertEqual(fallback.cookies.get("sessionid"), "fake-api-session")
        for cookie in fallback.cookies:
            self.assertEqual(cookie.domain, "cvat.edufun.ai")
            self.assertTrue(cookie.secure)
        for call in fallback.request.call_args_list:
            self.assertEqual(call.args, ("GET", "https://cvat.edufun.ai/api/jobs/206/data"))
            self.assertEqual(call.kwargs["timeout"], (15, 120))
            self.assertIs(call.kwargs["verify"], True)
            self.assertIs(call.kwargs["allow_redirects"], False)
            self.assertEqual(call.kwargs["params"]["quality"], "original")
        self.assertNotIn("fake-api-session", output.getvalue())
        self.assertNotIn("fake-api-csrf", output.getvalue())

    def test_pat_fallback_preserves_bearer_without_relogin(self):
        client, fallback = self.sdk_client(), self.fallback_session()
        client.sdk.cookies.clear()
        client.sdk.configuration.access_token = "fake-private-pat"
        client.sdk.rest_client.pool_manager.request = Mock(side_effect=read_error())
        with patch("cvat_client.requests.Session", return_value=fallback):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                client.frame_bytes(10, {10})
        self.assertEqual(fallback.headers["Authorization"], "Bearer fake-private-pat")
        self.assertEqual(fallback.request.call_count, 1)
        self.assertNotIn("fake-private-pat", output.getvalue())

    def test_http_403_and_certificate_failure_do_not_fallback(self):
        for response, error in ((raw_response(403, b'{"detail":"Permission denied"}'), None),
                                (None, ue.MaxRetryError(None, "/", ue.SSLError("private TLS message")))):
            with self.subTest(error=type(error).__name__):
                client = self.sdk_client()
                client.sdk.rest_client.pool_manager.request = Mock(return_value=response, side_effect=error)
                with patch("cvat_client.requests.Session") as session_type:
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(DiagnosticError):
                        client.frame_bytes(10, {10})
                    session_type.assert_not_called()

    def test_failed_fallback_is_not_repeated_and_clears_local_cookies(self):
        client, fallback = self.sdk_client(), self.fallback_session()
        client.sdk.rest_client.pool_manager.request = Mock(side_effect=read_error())
        fallback.request.side_effect = requests.exceptions.ReadTimeout("private")
        with patch("cvat_client.requests.Session", return_value=fallback):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(DiagnosticError) as caught:
                client.frame_bytes(10, {10})
        self.assertEqual(caught.exception.network_code, "READ_TIMEOUT")
        self.assertEqual(fallback.request.call_count, 1)
        self.assertEqual(len(fallback.cookies), 0)

    def test_outside_frame_stops_before_either_transport(self):
        client = self.sdk_client()
        client.sdk.rest_client.pool_manager.request = Mock()
        with patch("cvat_client.requests.Session") as session_type:
            with self.assertRaises(ValueError):
                client.frame_bytes(9, {10})
        client.sdk.rest_client.pool_manager.request.assert_not_called()
        session_type.assert_not_called()

    def test_post_login_and_patch_never_trigger_frame_fallback(self):
        from auth import Credentials
        for method in ("login", "patch"):
            with self.subTest(method=method):
                client = self.sdk_client()
                pool = Mock(side_effect=read_error())
                client.sdk.rest_client.pool_manager.request = pool
                with patch("cvat_client.requests.Session") as session_type:
                    with self.assertRaises(DiagnosticError):
                        if method == "login":
                            client.login(Credentials("password", "fake-user", "fake-password"))
                        else:
                            scope = Mock(job_id=206)
                            shape = {"frame": 10, "label_id": 7, "type": "rectangle", "points": [0., 0., 1., 1.],
                                     "source": "auto", "attributes": []}
                            client.create_shapes([shape], scope=scope, schema_verified=True, confirmed=True)
                self.assertEqual(pool.call_count, 1)
                session_type.assert_not_called()

    def test_rest_frame_and_metadata_use_separate_timeouts(self):
        client = CvatClient(Config(backend="rest", frame_timeout=180), Redactor())
        self.addCleanup(client.close)
        client.session.request = Mock(return_value=Mock(status_code=200, content=b"fake-image"))
        with contextlib.redirect_stdout(io.StringIO()):
            client.frame_bytes(10, {10})
        self.assertEqual(client.session.request.call_args.kwargs["timeout"], (15, 180))
        client.session.request.return_value = Mock(status_code=200, content=json.dumps({"version": "2.76.0"}).encode())
        with contextlib.redirect_stdout(io.StringIO()):
            client.about()
        self.assertEqual(client.session.request.call_args.kwargs["timeout"], 30)

    def test_frame_timeout_cli_bounds(self):
        config, _ = parse_config(["--frame-timeout", "180"])
        self.assertEqual(config.frame_timeout, 180)
        for value in (0, 601, float("nan")):
            with self.assertRaises(ValueError):
                Config(frame_timeout=value)


if __name__ == "__main__":
    unittest.main()
