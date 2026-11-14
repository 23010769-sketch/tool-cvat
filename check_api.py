"""Entry point chẩn đoán CVAT. Không tải ảnh hoặc chạy YOLO."""

from auth import DiagnosticError, Redactor, explain_status, read_credentials
from config import parse_config
from cvat_client import CvatClient
from permission_diagnostics import diagnose_permissions


def validate_scope(config, task, job, organization):
    if task.get("id") != config.task_id or job.get("id") != config.job_id:
        raise DiagnosticError("Phạm vi", None, "Server trả ID Task/Job khác yêu cầu; dừng.")
    if job.get("task_id") != config.task_id:
        raise DiagnosticError("Phạm vi", None, "Job không thuộc Task được chỉ định; dừng.")
    if config.organization:
        if organization.get("slug") != config.organization or organization.get("id") != task.get("organization"):
            raise DiagnosticError("Phạm vi", None, "Task không thuộc Organization được chỉ định; dừng.")
    elif task.get("organization") is not None or job.get("organization") is not None:
        raise DiagnosticError("Phạm vi", None, "Task/Job vẫn thuộc Organization; không phải Personal Workspace. Dừng.")
    if job.get("type") != "annotation":
        raise DiagnosticError("Phạm vi", None, "Chỉ cho phép Job loại annotation; dừng.")


def main(argv=None):
    config, args = parse_config(argv)
    redactor = Redactor()
    client = None
    user = None
    print("=" * 45)
    print("CVAT YOLO AUTO ANNOTATION — KIỂM TRA API")
    print(f"Server: {config.url}\nWorkspace: {config.organization or 'Personal Workspace'}")
    print(f"Task ID: {config.task_id}\nJob ID: {config.job_id}\nBackend: {config.backend}")
    print("Mode: READ ONLY — không chạy YOLO, không ghi annotation")
    try:
        client = CvatClient(config, redactor)
        about = client.about()
        version = about.get("version", "unknown")
        print("Server version:", redactor.clean(version))
        if version != "2.76.0":
            raise DiagnosticError("Phiên bản", None, "Cần kiểm tra lại API vì server khác CVAT SDK 2.76.0.")
        if args.public_only:
            print("Chỉ xác minh endpoint công khai; chưa kiểm tra xác thực hoặc quyền truy cập.")
            return 0
        print("[1/5] Checking authentication...")
        credentials = read_credentials(config.auth, args.username, redactor)
        client.login(credentials)
        user = client.current_user()
        print(f"[OK] Đã xác minh tài khoản ID {user['id']}.")
        client.set_organization()
        if args.inspect_annotations:
            from annotation_inventory import inspect_annotations
            inspect_annotations(client, config, redactor)
            return 0
        if args.diagnose_permissions:
            return diagnose_permissions(client, config, user)
        print("[2/5] Checking Task and Job...")
        task = client.task()
        org_id = task.get("organization")
        organization = {}
        if config.organization:
            if not isinstance(org_id, int) or isinstance(org_id, bool):
                raise DiagnosticError("Organization", None, "Task không thuộc Organization; dừng kiểm tra.")
            organization = client.organization(org_id)
        elif org_id is not None:
            raise DiagnosticError("Workspace", None,
                                  "Task vẫn thuộc Organization; đổi workspace trên web không chuyển Task. Dừng.")
        job = client.job()
        validate_scope(config, task, job, organization)
        print("[3/5] Reading Task labels...")
        labels = client.labels()
        if not labels:
            raise DiagnosticError("Labels", None, "Task chưa có label; chưa thể ánh xạ YOLO.")
        for label in labels:
            print(f"  ID {label['id']} | {redactor.clean(label['name'])} | type={redactor.clean(label.get('type', 'any'))}")
        print("[4/5] Checking annotation read access...")
        annotations = client.annotations()
        print(f"Existing shapes: {len(annotations.get('shapes', []))}; tracks: {len(annotations.get('tracks', []))}")
        print("Quyền ghi annotation: CHƯA XÁC MINH. Quyền đọc không chứng minh quyền PATCH/create.")
        print("[5/5] Reading Job frame metadata...")
        metadata = client.metadata()
        print(f"Job start={job.get('start_frame')}, stop={job.get('stop_frame')}; "
              f"metadata frames={len(metadata.get('frames', []))}; "
              f"included_frames={'present' if metadata.get('included_frames') is not None else 'absent'}")
        print("[OK] Kiểm tra xác thực, phạm vi và quyền đọc thành công.")
        print("Chưa tải ảnh, chạy YOLO hoặc ghi annotation. Tiếp theo: tích hợp YOLO sau khi kiểm tra này thành công.")
        return 0
    except DiagnosticError as error:
        # Step labels are trusted local strings, not response bodies or credentials.
        print(f"[LỖI] Bước: {error.step} | HTTP {error.status or 'N/A'}")
        print(error.detail)
        print(explain_status(error.status, error.step))
        if user and error.status == 403 and error.step == f"Task {config.task_id}":
            try:
                diagnose_permissions(client, config, user, task_denied=True)
            except DiagnosticError as diagnostic:
                print(f"[DỪNG CHẨN ĐOÁN] {diagnostic.step} | HTTP {diagnostic.status or 'N/A'} | {diagnostic.detail}")
        print("Đã dừng kiểm tra; không tải frame hoặc ghi annotation.")
        return 1
    except (ValueError, EOFError, KeyboardInterrupt) as error:
        print("[DỪNG]", redactor.clean(error) or "Đã hủy nhập thông tin.")
        return 2
    except Exception as error:
        print(f"[LỖI] {type(error).__name__}: dừng an toàn; không in exception có thể chứa bí mật.")
        return 1
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
