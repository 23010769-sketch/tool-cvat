"""Phân loại lỗi lồng nhau mà không in URL, headers, cookie hoặc exception message."""

import socket
import ssl

import requests
from urllib3 import exceptions as ue


def exception_chain(error):
    pending, result, seen = [error], [], set()
    while pending and len(result) < 16:
        item = pending.pop(0)
        if not isinstance(item, BaseException) or id(item) in seen:
            continue
        seen.add(id(item))
        result.append(item)
        pending.extend([getattr(item, "reason", None), item.__cause__, item.__context__])
        pending.extend(arg for arg in item.args if isinstance(arg, BaseException))
    return result


def describe_transport_error(error, timeout):
    chain = exception_chain(error)
    read_timeout = timeout[1] if isinstance(timeout, tuple) else timeout
    connect_timeout = timeout[0] if isinstance(timeout, tuple) else timeout

    def has(*types):
        return any(isinstance(item, types) for item in chain)

    if has(ssl.SSLCertVerificationError, ssl.CertificateError):
        code, message = "TLS_CERTIFICATE", "Không xác minh được chứng chỉ TLS; kiểm tra chuỗi CA/chứng chỉ server."
    elif has(ue.SSLError, requests.exceptions.SSLError):
        code, message = "TLS", "Lỗi TLS; không tắt xác minh SSL."
    elif has(ue.ProxyError, requests.exceptions.ProxyError):
        code, message = "PROXY", "Kết nối qua proxy thất bại; kiểm tra cấu hình proxy của Terminal."
    elif has(ue.NameResolutionError, socket.gaierror):
        code, message = "DNS", "Không phân giải được DNS của máy chủ."
    elif any(getattr(item, "winerror", None) == 10013 or getattr(item, "errno", None) in (13, 10013) for item in chain):
        code, message = "SOCKET_BLOCKED", "Hệ điều hành chặn socket; kiểm tra firewall/quyền kết nối (có thể WinError 10013)."
    elif has(ue.ReadTimeoutError, requests.exceptions.ReadTimeout):
        code, message = "READ_TIMEOUT", f"Quá thời gian chờ dữ liệu ({read_timeout:g}s giữa các lần đọc); chưa kết luận server lỗi hay xử lý chậm."
    elif has(ue.ConnectTimeoutError, requests.exceptions.ConnectTimeout) and not has(ue.NewConnectionError):
        code, message = "CONNECT_TIMEOUT", f"Quá thời gian thiết lập kết nối ({connect_timeout:g}s)."
    elif has(ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
        code, message = "CONNECTION_RESET", "Kết nối bị đóng/reset trước khi nhận đủ dữ liệu."
    elif has(ue.ProtocolError, ue.IncompleteRead, requests.exceptions.ChunkedEncodingError):
        code, message = "PROTOCOL", "Luồng HTTP bị ngắt hoặc chưa nhận đủ nội dung ảnh."
    elif has(ue.NewConnectionError, requests.exceptions.ConnectionError, ConnectionRefusedError):
        code, message = "CONNECTION", "Không thiết lập hoặc duy trì được kết nối tới máy chủ."
    elif has(TimeoutError, requests.exceptions.Timeout):
        code, message = "TIMEOUT", "Timeout khi kết nối/đọc dữ liệu; chưa xác định giai đoạn."
    else:
        code, message = "UNKNOWN", "Lỗi kết nối hoặc SDK chưa phân loại được."
    classes = " -> ".join(type(item).__name__ for item in chain)
    return code, f"[{code}] {message} Exception types: {classes}. URL/headers/nội dung exception đã ẩn."
