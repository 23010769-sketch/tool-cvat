"""Chẩn đoán 403 bằng các GET chính thức, giữ nguyên phạm vi và credentials."""

from auth import DiagnosticError


def diagnose_permissions(client, config, user, task_denied=False):
    print("--- CHẨN ĐOÁN QUYỀN: giữ nguyên tài khoản và workspace ---")

    def attempt(call):
        try:
            return call()
        except DiagnosticError as error:
            # Authentication loss, rate limits and network failures stop all probes.
            if error.status not in (403, 404):
                raise
            # step is built locally from endpoint names and configured numeric IDs.
            # Do not secret-filter these trusted names: a short secret may match an ID.
            print(f"[TỪ CHỐI] {error.step}: HTTP {error.status} | {error.detail}")
            return None

    organizations = attempt(client.organizations) if config.organization else []
    organization = next((org for org in organizations or []
                         if org.get("slug") == config.organization), None)
    if organization:
        print(f"Organization tìm thấy: ID {organization['id']}; slug={config.organization}")
    elif config.organization:
        print("Chưa xác minh được Organization qua danh sách; không đủ bằng chứng kết luận thiếu membership.")

    memberships = None
    if config.organization and user.get("username"):
        memberships = attempt(lambda: client.memberships(user["username"]))
    elif config.organization:
        print("Không có username trong /users/self để lọc membership; bỏ qua bước này.")
    matching = [m for m in memberships or [] if m.get("user", {}).get("id") == user["id"]
                and organization and m.get("organization") == organization["id"]]
    for membership in matching:
        # Role is untrusted server text and still needs secret filtering.
        role = client.redactor.clean(membership.get("role", "unknown"))
        print(f"Membership: role={role}; is_active={membership.get('is_active') is True}")
    if memberships is not None and not matching:
        print("Không có membership khớp được xác minh; nhờ quản trị viên kiểm tra membership/lời mời.")

    task = None if task_denied else attempt(client.task)
    task_ok = task is not None
    if task_ok:
        if task.get("id") != config.task_id:
            raise DiagnosticError("Phạm vi Task", None, "Task ID không khớp; dừng.")
        if config.organization:
            if not organization or task.get("organization") != organization["id"]:
                raise DiagnosticError("Phạm vi Task", None, "Chưa xác minh Task thuộc Organization yêu cầu; dừng.")
        elif task.get("organization") is not None:
            raise DiagnosticError("Phạm vi Task", None, "Task vẫn thuộc Organization, không phải Personal Workspace; dừng.")

    # Job is a separate authorized resource. A Task 403 does not establish Job permissions.
    job = attempt(client.job)
    if job is None:
        print(f"[KẾT LUẬN] Chưa đọc được Job {config.job_id}. Nhờ quản trị viên kiểm tra assignment cho user ID {user['id']}.")
        return 1
    if (job.get("id") != config.job_id or job.get("task_id") != config.task_id
            or job.get("type") != "annotation"):
        raise DiagnosticError("Phạm vi Job", None, "Job không khớp ID/Task/loại annotation yêu cầu; dừng.")
    org_id = job.get("organization")
    if config.organization:
        if not isinstance(org_id, int) or isinstance(org_id, bool):
            raise DiagnosticError("Phạm vi Job", None, "Job không có Organization ID hợp lệ; dừng.")
        if organization:
            if org_id != organization["id"]:
                raise DiagnosticError("Phạm vi Job", None, "Job thuộc Organization khác; dừng.")
        else:
            organization = attempt(lambda: client.organization(org_id))
            if (not organization or organization.get("id") != org_id
                    or organization.get("slug") != config.organization):
                raise DiagnosticError("Phạm vi Job", None, "Không xác minh được Organization thực của Job; dừng.")
    elif org_id is not None:
        raise DiagnosticError("Phạm vi Job", None, "Job vẫn thuộc Organization, không phải Personal Workspace; dừng.")

    assignee = job.get("assignee") or {}
    print(f"Job {config.job_id} thuộc Task {config.task_id}, đúng workspace đã chọn.")
    print(f"Job assignee ID: {assignee.get('id', 'none')}; là tài khoản hiện tại: {assignee.get('id') == user['id']}")
    labels = attempt(client.job_labels)
    annotations = attempt(client.annotations)
    metadata = attempt(client.metadata)
    print(f"Quyền đọc Job labels={labels is not None}; annotations={annotations is not None}; metadata={metadata is not None}")
    if not task_ok:
        print("[KẾT LUẬN] Đọc được Job nhưng chưa đọc được Task. Có thể do chỉ được giao Job; cần đối chiếu assignment/vai trò.")
    elif labels is not None and annotations is not None and metadata is not None:
        print("[KẾT LUẬN] Đã xác minh các quyền đọc được kiểm tra.")
    print("Chẩn đoán không xác minh quyền ghi; không tải ảnh, chạy YOLO hoặc sửa annotation.")
    return 0 if task_ok and labels is not None and annotations is not None and metadata is not None else 1
