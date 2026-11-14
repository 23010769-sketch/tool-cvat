"""Nhập bí mật tại Terminal và lọc thông báo lỗi trước khi in."""

import getpass
import json
import os
import re
import sys
from dataclasses import dataclass, field


class Redactor:
    def __init__(self):
        self._secrets = set()

    def add(self, *values):
        self._secrets.update(str(value) for value in values if value)

    def clean(self, value):
        text = str(value)
        for secret in sorted(self._secrets, key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
        text = re.sub(r"(?i)\b(Bearer|Token|Basic)\s+[^\s\"',;]+", r"\1 [REDACTED]", text)
        text = re.sub(r"(?i)\b(password|token|sessionid|csrftoken|cookie|authorization)\s*[:=]\s*[^\s,;]+",
                      r"\1=[REDACTED]", text)
        return "".join(c for c in text if c.isprintable())[:1200]

    def body(self, raw):
        try:
            data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        except (ValueError, UnicodeError):
            return "Phản hồi không phải JSON; bỏ nội dung HTML/text để tránh lộ dữ liệu."

        def scrub(obj):
            if isinstance(obj, dict):
                return {self.clean(k): "[REDACTED]" if re.search(
                    r"password|token|cookie|authorization|session|secret|email|username|key", k, re.I
                ) else scrub(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [scrub(v) for v in obj[:10]]
            return self.clean(obj)

        return self.clean(json.dumps(scrub(data), ensure_ascii=False))


@dataclass(repr=False)
class Credentials:
    method: str
    username: str = field(default="", repr=False)
    password: str = field(default="", repr=False)
    token: str = field(default="", repr=False)


def read_credentials(method, username, redactor):
    if method == "auto":
        method = "pat" if os.getenv("CVAT_ACCESS_TOKEN") else "password"
    if method == "pat":
        token = os.getenv("CVAT_ACCESS_TOKEN") or _secret("Personal Access Token: ")
        if not token.strip():
            raise ValueError("Personal Access Token không được trống.")
        redactor.add(token)
        return Credentials(method, token=token)
    username = username or os.getenv("CVAT_USERNAME")
    if not username:
        if not sys.stdin.isatty():
            raise ValueError("Chạy trong Terminal tương tác để nhập username/password; không gửi bí mật vào chat.")
        username = input("Username CVAT: ").strip()
    password = os.getenv("CVAT_PASSWORD") or _secret("Password CVAT (ẩn): ")
    if not username.strip() or not password:
        raise ValueError("Username/password không được trống.")
    redactor.add(username, password)
    return Credentials(method, username=username, password=password)


def _secret(prompt):
    if not sys.stdin.isatty():
        raise ValueError("Cần Terminal tương tác để nhập bí mật bằng getpass.")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            return getpass.getpass(prompt)
        except getpass.GetPassWarning:
            raise ValueError("Terminal không hỗ trợ nhập bí mật ẩn; hãy dùng PowerShell trong VS Code.") from None


class DiagnosticError(Exception):
    def __init__(self, step, status, detail, *, network_code=None):
        self.step, self.status, self.detail = step, status, detail
        self.network_code = network_code
        super().__init__(f"{step}: HTTP {status or 'N/A'} | {detail}")


def explain_status(status, step):
    if status == 400:
        return ("Server từ chối dữ liệu yêu cầu. Đọc lỗi validation bên trên: sai thông tin, "
                "thiếu trường, email chưa xác minh hoặc chính sách đăng nhập. HTTP 400 riêng lẻ không xác định nguyên nhân.")
    if status == 401:
        return "Thông tin xác thực không hợp lệ, hết hạn hoặc phương thức xác thực chưa được server chấp nhận."
    if status == 403:
        if step != "Đăng nhập":
            return ("Server từ chối quyền đối với endpoint này. Nếu /users/self đã thành công, "
                    "kiểm tra Organization, vai trò và Task/Job assignment. "
                    "Quyền đọc Task và quyền đọc Job cần được kiểm tra độc lập.")
        return ("Server từ chối truy cập. Khi /users/self đã thành công, kiểm tra membership Organization, "
                "vai trò, Task/Job assignment và phạm vi PAT. Tại login: kiểm tra lỗi CSRF/Origin hoặc chính sách server. "
                "Chỉ kết luận CSRF nếu phản hồi nêu rõ.")
    if status == 404:
        return "Không tìm thấy tài nguyên hoặc server ẩn tài nguyên do quyền truy cập; kiểm tra ID và Organization."
    if status == 429:
        return "Server giới hạn số lần gọi. Dừng; không tự lặp đăng nhập."
    if status and 300 <= status < 400:
        return "Endpoint chuyển hướng; đã dừng để không gửi bí mật sang URL khác. Kiểm tra URL/SSO với quản trị viên."
    return "Kiểm tra phản hồi server và kết nối TLS/DNS/firewall. Không tắt xác minh SSL."
