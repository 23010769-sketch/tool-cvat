"""SDK 2.76.0 ưu tiên; REST đối chiếu theo lựa chọn rõ ràng.

Chẩn đoán chỉ đọc; auto annotation dùng PATCH/create. Chuyển geometry riêng
chỉ dùng PATCH/update các ID đã xem trước và được xác nhận.
"""

import json
from http.cookies import SimpleCookie
from urllib.parse import urlsplit

import requests

from auth import DiagnosticError
from network_diagnostics import describe_transport_error


class CvatClient:
    def __init__(self, config, redactor):
        self.config, self.redactor = config, redactor
        self.session = None
        self.sdk = None
        self._frame_session = None
        if config.backend == "sdk":
            from cvat_sdk.api_client import ApiClient, Configuration
            from urllib3.util.retry import Retry
            sdk_config = Configuration(host=config.url.rstrip("/"), verify_ssl=True)
            sdk_config.retries = Retry(total=0, redirect=0, raise_on_redirect=False)
            self.sdk = ApiClient(sdk_config)
        else:
            self.session = requests.Session()
            self.session.headers.update({"Accept": "application/vnd.cvat+json, application/json"})

    def close(self):
        if self._frame_session:
            self._frame_session.cookies.clear()
            self._frame_session.close()
        if self.sdk:
            self.sdk.rest_client.pool_manager.clear()
            self.sdk.cookies.clear()
            self.sdk.close()
        if self.session:
            self.session.cookies.clear()
            self.session.close()

    def _cookies(self):
        if self.sdk:
            return {k: v.value for k, v in self.sdk.cookies.items()}
        return {cookie.name: cookie.value for cookie in self.session.cookies}

    def _call(self, step, sdk_call, method, path, *, binary=False, request_timeout=None,
              session_override=None, **kwargs):
        effective_timeout = self.config.timeout if request_timeout is None else request_timeout
        try:
            if self.sdk and session_override is None:
                _, response = sdk_call(_parse_response=False, _request_timeout=effective_timeout,
                                       _check_status=False)
                try:
                    status, raw = response.status, response.data
                    # SDK skips its cookie update when _parse_response=False.
                    # Keep API-issued cookies while parsing raw JSON ourselves.
                    headers = response.headers
                    cookie_headers = (headers.getlist("Set-Cookie") if hasattr(headers, "getlist")
                                      else [headers.get("Set-Cookie", "")])
                    for header in cookie_headers:
                        self.sdk.cookies.update(SimpleCookie(header))
                finally:
                    response.release_conn()
            else:
                session = session_override if session_override is not None else self.session
                response = session.request(method, self.config.url.rstrip("/") + path,
                                                timeout=effective_timeout, allow_redirects=False,
                                                verify=True, **kwargs)
                try:
                    status, raw = response.status_code, response.content
                    self.redactor.add(*(cookie.value for cookie in session.cookies))
                finally:
                    response.close()
            self.redactor.add(*self._cookies().values())
            if not 200 <= status < 300:
                raise DiagnosticError(step, status, self.redactor.body(raw))
            if binary:
                return raw
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeError):
                raise DiagnosticError(step, status, "Server không trả JSON hợp lệ; kiểm tra URL/proxy/SSO.") from None
            print(f"[OK] {step}: HTTP {status}")
            return data
        except DiagnosticError:
            raise
        except Exception as error:
            status = getattr(error, "status", None)
            # Never print a raw exception: SDK exceptions include headers/cookies.
            body = getattr(error, "body", None)
            code, summary = describe_transport_error(error, effective_timeout)
            detail = self.redactor.body(body) if body else summary
            raise DiagnosticError(step, status, detail, network_code=code if not body else None) from None

    def about(self):
        return self._call("Phiên bản server", self.sdk.server_api.retrieve_about if self.sdk else None,
                          "GET", "/api/server/about")

    def login(self, credentials):
        if credentials.method == "pat":
            if self.sdk:
                self.sdk.configuration.access_token = credentials.token
            else:
                self.session.headers["Authorization"] = "Bearer " + credentials.token
            print("[INFO] Đã cấu hình PAT; chưa kết luận xác thực đến khi /users/self thành công.")
            return
        payload = {"username": credentials.username, "password": credentials.password}
        sdk_call = None
        if self.sdk:
            from cvat_sdk.api_client import models
            sdk_call = lambda **kw: self.sdk.auth_api.create_login(models.LoginSerializerExRequest(**payload), **kw)
        result = self._call("Đăng nhập", sdk_call, "POST", "/api/auth/login", json=payload)
        if isinstance(result, dict):
            self.redactor.add(result.get("key"), result.get("token"), result.get("access_token"))
        cookies = self._cookies()
        if not cookies.get("sessionid") or not cookies.get("csrftoken"):
            raise DiagnosticError("Session authentication", None,
                                  "Login không cấp đủ sessionid/csrftoken. Dừng; thử PAT nếu server hỗ trợ.")
        origin = self.config.url.rstrip("/")
        if self.sdk:
            self.sdk.configuration.api_key["csrfHeaderAuth"] = cookies["csrftoken"]
            self.sdk.set_default_header("Origin", origin)
        else:
            self.session.headers.update({"X-CSRFToken": cookies["csrftoken"], "Origin": origin})

    def current_user(self):
        data = self._call("Tài khoản /users/self", self.sdk.users_api.retrieve_self if self.sdk else None,
                          "GET", "/api/users/self")
        if not isinstance(data, dict) or not isinstance(data.get("id"), int):
            raise DiagnosticError("Tài khoản", None, "Phản hồi tài khoản không có user ID hợp lệ.")
        return data

    def set_organization(self):
        headers = self.sdk.default_headers if self.sdk else self.session.headers
        if self.config.organization:
            headers["X-Organization"] = self.config.organization
        else:
            headers.pop("X-Organization", None)

    def task(self):
        task_id = self.config.task_id
        call = (lambda **kw: self.sdk.tasks_api.retrieve(task_id, **kw)) if self.sdk else None
        return self._call(f"Task {task_id}", call, "GET", f"/api/tasks/{task_id}")

    def organization(self, org_id):
        call = (lambda **kw: self.sdk.organizations_api.retrieve(org_id, **kw)) if self.sdk else None
        return self._call("Organization của Task", call, "GET", f"/api/organizations/{org_id}")

    def job(self):
        job_id = self.config.job_id
        call = (lambda **kw: self.sdk.jobs_api.retrieve(job_id, **kw)) if self.sdk else None
        return self._call(f"Job {job_id}", call, "GET", f"/api/jobs/{job_id}")

    def labels(self):
        api = self.sdk.labels_api if self.sdk else None
        return self._list("Labels", api, "/api/labels", {"task_id": self.config.task_id})

    def job_labels(self):
        api = self.sdk.labels_api if self.sdk else None
        return self._list("Labels của Job", api, "/api/labels", {"job_id": self.config.job_id})

    def organizations(self):
        api = self.sdk.organizations_api if self.sdk else None
        return self._list("Organization theo slug", api, "/api/organizations",
                          {"slug": self.config.organization})

    def memberships(self, username):
        api = self.sdk.memberships_api if self.sdk else None
        # CVAT 2.76.0 maps the user equality filter to user__username.
        return self._list("Membership của tài khoản", api, "/api/memberships",
                          {"user": username, "org": self.config.organization})

    def _list(self, step, api, path, filters):
        result, page = [], 1
        while True:
            params = {**filters, "page": page, "page_size": 100}
            call = (lambda **kw: api.list(**params, **kw)) if api else None
            data = self._call(f"{step} trang {page}", call, "GET", path, params=params)
            if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                raise DiagnosticError(step, None, "Cấu trúc phân trang không hợp lệ.")
            result.extend(data["results"])
            if not data.get("next"):
                return result
            if not data["results"] or page >= 1000:
                raise DiagnosticError(step, None, "Phân trang bất thường; đã dừng.")
            page += 1

    def annotations(self):
        job_id = self.config.job_id
        call = (lambda **kw: self.sdk.jobs_api.retrieve_annotations(job_id, **kw)) if self.sdk else None
        return self._call("Quyền đọc annotation", call, "GET", f"/api/jobs/{job_id}/annotations")

    def metadata(self):
        job_id = self.config.job_id
        call = (lambda **kw: self.sdk.jobs_api.retrieve_data_meta(job_id, **kw)) if self.sdk else None
        return self._call("Metadata frame của Job", call, "GET", f"/api/jobs/{job_id}/data/meta")

    def frame_bytes(self, frame, allowed_frames):
        if type(frame) is not int or frame not in allowed_frames:
            raise ValueError("Frame ngoài phạm vi Job đã xác minh.")
        job_id = self.config.job_id
        params = {"type": "frame", "number": frame, "quality": "original"}
        call = (lambda **kw: self.sdk.jobs_api.retrieve_data(job_id, **params, **kw)) if self.sdk else None
        timeout = (min(self.config.timeout, 15), self.config.frame_timeout)
        print(f"[INFO] Đọc Job {job_id}, frame {frame}, quality=original; read timeout={self.config.frame_timeout:g}s")
        try:
            return self._call("Đọc frame của Job", call, "GET", f"/api/jobs/{job_id}/data",
                              binary=True, params=params, request_timeout=timeout,
                              session_override=self._frame_session)
        except DiagnosticError as error:
            retryable = {"READ_TIMEOUT", "CONNECT_TIMEOUT", "TIMEOUT", "CONNECTION_RESET", "PROTOCOL", "CONNECTION", "PROXY"}
            if not self.sdk or self._frame_session is not None or error.status or error.network_code not in retryable:
                raise
            print(f"[INFO] SDK tải frame thất bại: {error.detail}")
            print("[INFO] Đối chiếu GET frame bằng requests một lần, dùng cùng phiên API và Organization; không đăng nhập lại.")
            session = requests.Session()
            session.headers.update({key: value for key, value in self.sdk.default_headers.items()
                                    if key.casefold() not in ("cookie", "host", "content-type")})
            session.headers["Accept"] = "image/*, application/octet-stream"
            token = self.sdk.configuration.access_token
            if token:
                session.headers["Authorization"] = "Bearer " + token
            host = urlsplit(self.config.url).hostname
            for name, value in self._cookies().items():
                session.cookies.set(name, value, domain=host, path="/", secure=True)
            try:
                raw = self._call("Đọc frame của Job bằng REST", None, "GET", f"/api/jobs/{job_id}/data",
                                 binary=True, params=params, request_timeout=timeout, session_override=session)
            except Exception:
                session.cookies.clear()
                session.close()
                raise
            self._frame_session = session
            print("[OK] REST tải frame thành công; dùng kết nối này cho các frame tiếp theo trong lần chạy.")
            return raw

    def schema(self):
        # Content negotiation is required; format=json alone can still return YAML.
        call = self.sdk.schema_api.retrieve if self.sdk else None
        raw = self._call("OpenAPI schema", call, "GET", "/api/schema/", binary=True,
                         headers={"Accept": "application/vnd.oai.openapi+json"})
        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or "openapi" not in data:
                raise ValueError("not OpenAPI")
            return data
        except Exception:
            raise DiagnosticError("OpenAPI schema", None, "Không đọc được schema chính thức; chặn ghi annotation.") from None

    def update_shape_geometry(self, shapes, *, originals, scope, schema_verified, confirmed):
        """Explicit format conversion only; never used by normal auto annotation."""
        if not confirmed or not schema_verified or scope.job_id != self.config.job_id or not shapes:
            raise ValueError("Chưa xác nhận chuyển định dạng/phạm vi/schema.")
        from box_conversion import validate_update
        from cvat_sdk.api_client import models
        for shape in shapes:
            validate_update(shape, originals.get(shape.get("id")), scope)
        body = {"tags": [], "tracks": [], "intervals": [], "shapes": shapes}
        call = None
        if self.sdk:
            payload = models.PatchedLabeledDataRequest(**body)
            call = lambda **kw: self.sdk.jobs_api.partial_update_annotations(
                "update", self.config.job_id, patched_labeled_data_request=payload, **kw)
        try:
            return self._call("Đổi geometry thành box", call, "PATCH", f"/api/jobs/{self.config.job_id}/annotations/",
                              params={"action": "update"}, json=body)
        except DiagnosticError as error:
            if not error.status or error.status >= 500 or 200 <= error.status < 300:
                error.detail += " Kết quả cập nhật chưa rõ; không gửi lại. Đọc lại annotation trước lần chạy sau."
            raise

    def create_shapes(self, shapes, *, scope, schema_verified, confirmed):
        # Defense in depth: callers cannot select task endpoints or a different Job.
        if not schema_verified or not confirmed or scope.job_id != self.config.job_id:
            raise ValueError("Chưa xác minh schema/phạm vi hoặc chưa xác nhận ghi.")
        for shape in shapes:
            scope.validate_shape(shape)
        if not shapes:
            raise ValueError("Không gửi PATCH rỗng để thử quyền.")
        from cvat_sdk.api_client import models
        body = {"tags": [], "tracks": [], "intervals": [], "shapes": shapes}
        call = None
        if self.sdk:
            payload = models.PatchedLabeledDataRequest(**body)
            call = lambda **kw: self.sdk.jobs_api.partial_update_annotations(
                "create", self.config.job_id, patched_labeled_data_request=payload, **kw)
        try:
            return self._call("Thêm shapes vào Job", call, "PATCH", f"/api/jobs/{self.config.job_id}/annotations/",
                              params={"action": "create"}, json=body)
        except DiagnosticError as error:
            if not error.status or error.status >= 500 or 200 <= error.status < 300:
                error.detail += " Kết quả ghi chưa rõ; không tự gửi lại. Đọc lại annotation trước lần chạy sau."
            raise
