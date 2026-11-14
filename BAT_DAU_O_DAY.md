# Mở lại dự án CVAT trong thư mục 3107

- Nhấp đúp **MO_DU_AN.cmd** để mở cả thư mục bằng VS Code.
- Nhấp đúp **MO_TERMINAL.cmd** để mở Terminal tại đây và xem trước bản nhãn. Bước mở này không ghi lên CVAT.
- Mở Job trên web: https://cvat.edufun.ai/tasks/211/jobs/908

Thư mục có đủ mã nguồn, môi trường Python hiện tại, model, dữ liệu 20 ảnh,
mask/plan, bản sao nhãn và các file guideline. Môi trường kèm theo được kiểm tra
để dùng trên chính máy Windows này, với bản Python 3.14 đã cài.

Gán bản mask đã chuẩn bị khi cần:

```powershell
.\.venv\Scripts\python.exe .\annotate_job908.py --apply
```

Bỏ vùng gạch đỏ từ các Issue tự động:

```powershell
.\.venv\Scripts\python.exe .\annotate_job908.py --remove-auto-issues
```

Nhập tài khoản và mật khẩu ẩn tại Terminal khi được hỏi. Mật khẩu không được lưu.
Ghi chú review: `outputs/job908_semantic/README_REVIEW.md`.

File **cvat-yolo-3107.zip** là bản sao nén của dự án này. Giải nén toàn bộ để
lấy lại đầy đủ các file; không chỉ lấy riêng `annotate_job908.py`.
