"""Reviewed semantic repair for Task 211 / Job 908; login stays in Terminal."""

import argparse
from redo_job908 import main as annotation_main


def main(argv=None):
    parser = argparse.ArgumentParser(description="Task 211 / Job 908: mask semantic đã chuẩn bị; mặc định chỉ xem trước.")
    parser.add_argument("--redo-guidelines", action="store_true", help="Tương thích lệnh cũ; sửa theo guideline hiện là chế độ mặc định.")
    parser.add_argument("--apply", action="store_true", help="Đăng nhập Terminal và thay box tự động cũ bằng mask semantic đã chuẩn bị.")
    parser.add_argument("--backend", choices=("rest",), default="rest")
    parser.add_argument("--auth", choices=("auto", "password", "pat", "session"), default="password")
    parser.add_argument("--frame-timeout", type=float, default=180)
    parser.add_argument("--audit-labels", action="store_true", help="Chỉ kiểm tra labels, không tải frame hoặc ghi.")
    parser.add_argument("--prepare", action="store_true", help="Chuẩn bị lại bản sửa từ các mask đã được review.")
    parser.add_argument("--remove-auto-issues", action="store_true", help="Bỏ gạch đỏ từ Issue tự động, giữ nguyên nhãn mask.")
    args = parser.parse_args(argv)
    if args.apply and args.audit_labels:
        parser.error("--audit-labels chạy riêng, không kết hợp --apply.")
    if args.audit_labels and args.prepare:
        parser.error("--audit-labels chạy riêng, không kết hợp --prepare.")
    if args.remove_auto_issues and (args.apply or args.prepare or args.audit_labels):
        parser.error("--remove-auto-issues chạy riêng.")
    options = ["--auth", args.auth, "--frame-timeout", str(args.frame_timeout)]
    if args.audit_labels:
        options.append("--audit-labels")
    if args.prepare:
        options.append("--prepare")
    if args.apply:
        options.append("--apply")
    if args.remove_auto_issues:
        options.append("--remove-auto-issues")
    return annotation_main(options)


if __name__ == "__main__":
    raise SystemExit(main())
