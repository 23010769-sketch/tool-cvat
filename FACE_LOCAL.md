# Gán 13 điểm mặt ở CVAT local

`annotate_face_local.py` chỉ làm Task 5 / Job 5, lấy tên và ID nhãn từ CVAT
hiện tại. Model MediaPipe được nạp từ `models/face_landmarker.task`, kiểm tra
SHA-256 trước khi dùng. Script không tự tải model, không mở trình duyệt và
không gửi ảnh hoặc tọa độ vào cuộc trò chuyện. Kết quả thật, bản sao nhãn và
ảnh xem trước chỉ lưu ở `outputs/local_pose/job5_private/`.

Trong phiên trợ lý, bước xử lý ảnh chạy trong sandbox chặn kết nối Internet.
Bước tải bộ cài/model công khai được thực hiện riêng, trước khi đọc ảnh.
Khi tự chạy ngoài sandbox, hãy dùng môi trường đã được đơn vị chấp thuận và
ngắt Internet nhưng giữ CVAT localhost hoạt động. MediaPipe có cơ chế gửi
metrics khi có mạng; script Python không phải là tường lửa cho thư viện native.

Các điểm theo trái/phải giải phẫu của người trong ảnh:

- Hai mắt: khóe ngoài, khóe trong, giữa mí trên, giữa mí dưới (8 điểm).
- Đầu mũi (1 điểm).
- Hai khóe miệng, giữa bờ môi trên và bờ môi dưới phía ngoài (4 điểm).

Nhãn có sẵn được giữ nguyên; người đã có nhãn được so khớp để tránh thêm trùng.
Nếu có thay đổi nhãn/ảnh không dự kiến hoặc ghi bị gián đoạn, công cụ dừng để
đối chiếu. Chỉ PATCH/create, không sửa hoặc xóa annotation cũ. Sau mỗi lần ghi,
đọc lại xác minh 13 điểm và kiểm tra annotation cũ còn nguyên. Không đổi stage/state.

Chạy xem trước trong môi trường không có Internet:

```powershell
.\.venv-face\Scripts\python.exe .\annotate_face_local.py
```

Ghi bản đã chuẩn bị và kiểm tra trực tiếp trong CVAT:

```powershell
.\.venv-face\Scripts\python.exe .\annotate_face_local.py --apply
```

Đăng nhập qua `cvat_local_session.py --gui` nếu phiên kết nối cũ đã đóng.
Mật khẩu chỉ gửi tới CVAT localhost, giữ phiên trong bộ nhớ. Không đưa file
nhãn, ảnh xem trước hoặc file lỗi riêng tư vào chat. Console của công cụ gán
chỉ in trạng thái và số lượng tổng hợp. Điểm do model dự đoán vẫn cần người dùng
rà trực tiếp trong CVAT, đặc biệt khi mặt nghiêng hoặc có phần bị che.

## Làm lại toàn bộ nhãn mặt khi người dùng yêu cầu

`redo_face_local.py` tạo bản mới độc lập với nhãn cũ. Các điểm giữa mí mắt
và giữa bờ môi được tính từ giao điểm đường viền với đường giữa hai khóe;
ba biến thể ảnh cục bộ được đối chiếu trước khi lấy trung vị. Nhãn cũ, metadata,
schema và các bước ghi được sao lưu trong thư mục `job5_private/redo_*`.

```powershell
.\.venv-face\Scripts\python.exe .\redo_face_local.py
.\.venv-face\Scripts\python.exe .\redo_face_local.py --apply
```

Chế độ apply thêm và đọc lại xác minh toàn bộ nhãn mới trước, rồi mới
PATCH/delete đúng ID của các skeleton `face` cũ đã sao lưu. Các annotation
khác được đối chiếu giữ nguyên. Nếu có lỗi hoặc dữ liệu thay đổi, công cụ dừng;
không tự lặp lại yêu cầu ghi chưa rõ kết quả. Khi chạy lại `--apply`, công cụ
đọc và đối chiếu tình trạng hiện tại trước khi quyết định bước còn thiếu.
