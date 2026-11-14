# CVAT YOLO Auto Annotation trên Windows

## Làm lại Job 908 theo guideline semantic

Ảnh `week1-seg/G04/G04_S081.jpg` đến `G04_S100.jpg` đã được đọc từ Job 908
(Task 211, Organization `cohort-5a`). Bản nháp dùng SegFormer cục bộ, có sửa trực
tiếp theo ảnh và hai lượt kiểm tra; không lấy ground truth/annotation BDD có sẵn.
226 mask semantic thay cho 191 rectangle tự động. Bản này còn 20 ghi chú review
theo frame về vùng tối, vật ngoài taxonomy và các quy ước chưa được guideline nêu
rõ (đặc biệt tuyết và phần xe gắn camera/capo). Chưa được xem là bài đã hoàn tất.

Xem trước bản sửa, không đăng nhập hoặc ghi CVAT:

```powershell
.\.venv\Scripts\python.exe .\annotate_job908.py
```

Ghi bản sửa đã chuẩn bị, nhập username và mật khẩu ẩn tại Terminal:

```powershell
.\.venv\Scripts\python.exe .\annotate_job908.py --apply
```

Chế độ này dùng REST và kiểm tra đúng tài khoản được giao Job, Stage annotation,
19 tên nhãn semantic, kích thước và từng pixel ảnh, cùng annotation cũ trước khi
ghi. Nó thêm mask, đọc lại xác minh đầy đủ rồi mới PATCH/delete đúng các ID box
tự động cũ trong bản sao. Nếu dữ liệu đã thay đổi hoặc kết quả ghi chưa rõ, nó dừng
để đối chiếu; không tự gửi lặp hoặc xóa nhãn vẽ tay. Chạy lại sau lần ghi dở sẽ nhận
diện trường hợp đã có đủ mask, hoặc đã thay xong, để tránh tạo trùng.

Ghi chú review được giữ trong `outputs/job908_semantic/README_REVIEW.md`, không
tự tạo Issue phủ gạch đỏ lên ảnh. Stage/State và cấu hình label không tự thay đổi.
Chỉ Finish Job sau khi đã kiểm tra chất lượng và xử lý những điểm chưa rõ.
Đầu ra xem trước: `outputs/job908_semantic/review_*.jpg`; bản sửa cố định và mã
kiểm tra: `outputs/job908_semantic/plan.json`. Trạng thái lưu nằm tại
`outputs/job908_semantic/apply_status.json`. Các file này không chứa mật khẩu.

Để bỏ các vùng gạch đỏ từ 20 Issue tự động đã tạo trước đây, đăng nhập Terminal:

```powershell
.\.venv\Scripts\python.exe .\annotate_job908.py --remove-auto-issues
```

Lệnh chỉ xóa Issue do chính tài khoản hiện tại tạo từ bản sửa Job 908 này, kiểm
tra nhãn trước/sau để giữ nguyên mask và giữ các Issue khác. Tải lại trang CVAT
sau khi báo `[OK]`. Lệnh `--apply` cũng dọn các Issue tự động cũ và không tạo lại.

Đối chiếu labels chỉ xét schema của 19 lớp semantic: ID, tên, khả năng ghi mask,
parent và thuộc tính. Màu, thứ tự trả về, URL và các label ngoài phạm vi semantic
không làm dừng việc ghi. Nếu schema thực sự khác, xem chi tiết tại
`outputs/job908_semantic/label_comparison.json`; phản hồi labels hiện tại được
lưu riêng ở `labels_live.json`, không ghi đè bản gốc đã review.

Thuộc tính mới có giá trị mặc định hợp lệ (ví dụ `occluded=false`,
`truncated=false`) được bổ sung vào payload mask bằng đúng spec ID hiện tại,
kiểm tra bằng validator của CVAT. Box cũ có thể trả thêm các giá trị mặc định
này mà không làm dừng thao tác; thay đổi giá trị thật hoặc hình học vẫn được
phát hiện. Pixel mask và bản sao gốc được giữ nguyên. Các cờ mặc định không
thay thế bước review occlusion/truncation cho từng đối tượng khi bài yêu cầu.

`annotate_job908.py` mặc định dùng bản semantic đã chuẩn bị; cờ
`--redo-guidelines` cũ vẫn được chấp nhận. Profile box `scene_job908.toml`
không phù hợp với bài semantic.

Chỉ đăng nhập kiểm tra 19 label semantic, không tải ảnh hoặc ghi:

```powershell
.\.venv\Scripts\python.exe .\annotate_job908.py --audit-labels
```

Báo cáo: `outputs/job908_semantic/semantic_label_audit.json`.

Các luồng YOLO/box cho Job 206 dưới đây dùng cấu hình riêng.

## Task 71 / Job 206

Theo yêu cầu mới, mọi kết quả đều dùng **rectangle như car**, kể cả road/sky/lane.
`scene_job206.toml` dùng YOLO + SegFormer để tạo box cho **22/31 labels**.
Mỗi vùng semantic rời đủ ngưỡng có box riêng, không gom nhiều cột/người thành một box.
`all_boxes_job206.toml` bổ sung 9 lớp lane/drivable từ annotation BDD gốc, chỉ khi
**tất cả ảnh Job giữ tên BDD và trùng từng pixel RGB với ảnh nguồn**. Đây là chuyển
nhãn tham chiếu, không phải model lane; ảnh trường không được upload ra ngoài.

Đổi các mask/đường tự động đã có sang box trước (không tải ảnh hoặc nạp model):

```powershell
.\.venv\Scripts\python.exe .\main.py --organization cohort-5a --auth password --convert-existing-to-boxes --all --apply
```

Lệnh xem trước và hỏi `CONVERT JOB 206 TO BOXES` cho các giai đoạn 1/5/toàn Job.
Giữ nguyên box cũ, tên label, frame, thuộc tính và ID gốc. Mask nhiều vùng được
tách box; vùng lớn nhất giữ ID gốc, các box còn lại được thêm trước khi đổi mask.
Không sửa nhãn vẽ tay hoặc track. Mặc định bỏ nhiễu dưới 64 pixel và giữ ít nhất
vùng lớn nhất để không làm mất nhãn. Không có PUT, DELETE hay Import/Export.
Đây là chế độ **đổi geometry đã tồn tại** được yêu cầu rõ ràng; luồng gán mới vẫn
chỉ PATCH/create, không tự chạy chuyển đổi. Bỏ `--apply` để chỉ xem trước.

Gán các nhãn mới theo cùng định dạng box, gồm lane/drivable nếu đối chiếu BDD đạt:

```powershell
.\.venv\Scripts\python.exe .\main.py --organization cohort-5a --auth password --config .\all_boxes_job206.toml --all --apply --frame-timeout 180
```

Nếu tên ảnh bị đổi, pixel bị sửa hoặc không có reference, lệnh dừng trước khi ghi;
không tự suy đoán lane. Khi đó vẫn có thể gán 22 lớp bằng profile scene dưới đây.
Chi tiết ID, nguồn và giới hạn: [LABELS_JOB_206.md](LABELS_JOB_206.md).

Để dùng pipeline mới trên tất cả ảnh của Job theo các bước thử và xác nhận ghi:

```powershell
.\.venv\Scripts\python.exe .\main.py --organization cohort-5a --auth password --config .\scene_job206.toml --all --apply --frame-timeout 180
```

Model mới dùng checkpoint riêng; không xóa checkpoint cũ. Mỗi bước ghi vẫn hỏi
`APPLY JOB 206`. Kiểm tra ảnh đã ghi thử trên CVAT trước khi xác nhận bước tiếp theo.
Bật biểu tượng con mắt của các label trên CVAT nếu chúng đang ẩn. Công cụ chưa
gửi yêu cầu ghi thật khi phát triển pipeline mới.

Các model scene và dependency đã được chuẩn bị trong workspace. Khi thiết lập lại:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-scene.txt
.\.venv\Scripts\python.exe .\prepare_bdd_model.py
.\.venv\Scripts\python.exe .\prepare_scene_model.py
```

`Created` là số box mới, không phải số ảnh. Xem `Processed frames`
để biết số ảnh đã xử lý. Box semantic được bao quanh vùng model dự đoán đủ confidence,
không từ việc một label tồn tại trong Job. Chương trình không bắt buộc mỗi ảnh có đủ mọi lớp.
Model, config và weights semantic được ghim checksum, chỉ nạp cục bộ. Ảnh không rời máy.

Đã xác nhận từ kết quả kiểm tra của tài khoản ID **212** ngày 10/11/2026:
Organization **cohort-5a (ID 1)**, membership `worker` đang hoạt động;
Job **206** thuộc Task **71** và được giao cho tài khoản này. Job, labels,
annotations và metadata đều trả HTTP 200. Đọc Task cha trả 403. Luồng annotation
vì vậy sử dụng các endpoint của **Job 206**, không yêu cầu quyền đọc Task cha.
Đây là quyền truy cập riêng theo chính sách CVAT, không phải vượt phân quyền.

Mã nguồn đã kiểm tra với server/SDK **2.76.0**, Python **3.14.3**, Ultralytics
**8.4.208**, Torch **2.15.0+cpu**. Weights nano `models/yolo11n.pt` đã tải từ
Ultralytics assets và chạy CPU thành công trên ảnh trống tự tạo. Người dùng đã chạy
detection trên ảnh thật và ghi thành công; kiểm kê có 97 rectangle. OpenAPI công khai
đã xác nhận PATCH Job/action=create cho rectangle và action=update có ID.
Luồng box kết hợp đã thử CPU trên một ảnh BDD công khai: 31 lớp được ánh xạ,
66 rectangle đầu ra, gồm 15 box lane và 6 biển báo. Chưa ghi thật bằng công cụ
trong phiên phát triển; user chạy và xác nhận tại Terminal để thao tác trên Job.

## Bắt đầu với 1 frame

Mở Terminal PowerShell trong VS Code tại `C:\Users\ADMIN\cvat-yolo`:

```powershell
.\.venv\Scripts\python.exe .\main.py --organization cohort-5a --auth password --limit 1
```

Nhập username và password tại Terminal; password không hiện ký tự. Không gửi bí mật
vào chat. Mỗi lần chạy đăng nhập lại; session/csrf chỉ được giữ trong RAM và do
API cấp, không đọc cookie trình duyệt.

Sau đăng nhập chương trình xác minh phiên bản, Job/Task ID, Organization thực,
labels, quyền đọc annotations và metadata trước khi nạp model hoặc tải frame.
Nó in mapping, lớp chưa hỗ trợ, frame ID, confidence và tọa độ pixel của đối tượng.
Nếu không có mapping hợp lệ, sẽ dừng trước khi lấy ảnh.

Sau khi dry-run 1 frame thành công:

```powershell
.\.venv\Scripts\python.exe .\main.py --organization cohort-5a --limit 5
# Sau khi dry-run 5 frame thành công mới chạy toàn Job:
.\.venv\Scripts\python.exe .\main.py --organization cohort-5a
```

Các chương trình mặc định Organization `cohort-5a` nếu chưa có biến môi trường cấu hình.
Các lệnh ghi rõ Organization ở trên vẫn ưu tiên hơn biến môi trường cũ. Không dùng
`--personal` cho Job 206 vì Job này thuộc Organization. Chọn Personal trên web
không chuyển Task/Job hoặc cấp thêm quyền.

## Gán tất cả ảnh của Job trong một lần đăng nhập

`Created: 5` nghĩa là thêm **5 bounding box**, không có nghĩa đã xử lý 5 ảnh.
`--limit 1` chỉ lấy ảnh đầu, `--limit 5` chỉ lấy tối đa 5 ảnh đầu.
Để hoàn tất toàn bộ Job, chạy:

```powershell
.\.venv\Scripts\python.exe .\main.py --organization cohort-5a --auth password --all --apply --frame-timeout 180
```

Chế độ `--all --apply` giữ một phiên đăng nhập, model và mapping trong suốt lần chạy.
Nó đọc checkpoint đúng tài khoản/phạm vi để hoàn tất các bước còn thiếu theo thứ tự:
dry-run 1 ảnh → dry-run 5 ảnh → ghi thử 1 ảnh → ghi thử 5 ảnh → dry-run toàn Job → ghi toàn Job.
Bước đã hoàn tất được bỏ qua. Với checkpoint hiện tại đã ghi 1 ảnh, chương trình
tiếp tục từ bước ghi 5 ảnh. Mỗi giai đoạn ghi đều xem trước và hỏi `APPLY JOB 206`.
Hủy xác nhận hoặc có lỗi sẽ dừng toàn bộ các bước còn lại.

Terminal hiển thị `[TOÀN JOB] N frame hợp lệ` và `[GIAI ĐOẠN] ... X/N frame`, với N
lấy từ API, không hardcode 20 hoặc tự đọc sang Job khác. Với Job có 20 ảnh hợp lệ,
giai đoạn toàn Job xử lý đủ 20 ảnh. Các box đã tạo được chống trùng và giữ nguyên.
Ảnh không có đối tượng đạt confidence/mapping có thể không có box mới; không tạo
nhãn giả chỉ để mọi ảnh đều có annotation.

`--all` khi không có `--apply` chỉ hoàn tất các bước dry-run còn thiếu và không ghi.
Không dùng chung `--all` với `--limit`. Các lệnh từng bước phía dưới vẫn hoạt động.

## Mapping và cấu hình YOLO

Danh sách 31 labels người dùng đã gửi được phân nhóm và ghi trạng thái thực tế tại
[LABELS_JOB_206.md](LABELS_JOB_206.md). Đã chuẩn bị thêm profile detection 10 lớp
BDD100K trong `bdd_detection.toml`, gồm rider và traffic sign. **Profile này chưa hỗ trợ
đủ 31 labels**. `scene_job206.toml` bổ sung 12 lớp semantic bằng box;
`all_boxes_job206.toml` hỗ trợ thêm drivable/lane khi ảnh trùng BDD đã xác minh.
`prepare_bdd_model.py` tải weights đã ghim revision và SHA-256, không kết nối CVAT.
Mặc định `annotation.toml` cũ được giữ nguyên để không âm thầm đổi model hoặc checkpoint.

`--all` nghĩa là **tất cả frame của Job**, không phải model hỗ trợ tất cả labels.
Để đối chiếu toàn bộ labels thực tế với lớp thật của weights và writer hiện tại:

```powershell
.\.venv\Scripts\python.exe .\main.py --organization cohort-5a --auth password --audit-labels
```

Lệnh này chỉ đăng nhập, xác minh Job và đọc labels, nạp model để lấy tên lớp;
không đọc frame, không chạy inference ảnh, không thay checkpoint và không ghi annotation.
Báo cáo metadata được lưu ở `outputs/job_206_label_audit.json`, chỉ gồm tên/ID/type,
lớp model và trạng thái hỗ trợ; không lưu ảnh, credentials hoặc annotations của ảnh.
Báo cáo `LABEL COVERAGE` hiển thị ID/tên/type và trạng thái cho từng label:
`supported`, `ambiguous` (cần chọn mapping), `not_selected`, `attributes` hoặc `unsupported`.
Sublabel của skeleton cũng được liệt kê. Không đổi tên một dự đoán person thành
mọi nhãn hoặc nhân cùng box sang các lớp khác để tuyên bố đã gán đủ labels.

Khi muốn yêu cầu hỗ trợ đủ labels trước khi chạy, thêm `--require-all-labels`
vào lệnh dry-run/apply. Cờ này dừng trước khi đọc ảnh hoặc ghi nếu còn label chưa
được model/mapping/geometry hỗ trợ; nó không tự tạo model cho các lớp còn thiếu.
Hỗ trợ đủ lớp không bảo đảm phát hiện đủ mọi đối tượng trong mọi ảnh.
Gửi phần `LABEL COVERAGE` (không có bí mật hoặc ảnh) để xác định cần mapping hay
weights/model riêng cho những nhãn còn thiếu.

Sửa `annotation.toml`:

```toml
[yolo]
model = "models/yolo11n.pt"
confidence = 0.45
iou = 0.50
duplicate_iou = 0.85
batch_size = 1
image_size = 640
device = "cpu"

[labels]
# Chỉ thêm nếu tên đích thực sự có trong Job và đúng nghĩa:
# person = "Pedestrian"
# motorcycle = "Motorbike"
```

Hỗ trợ label trùng tên model, không phân biệt hoa/thường và chuẩn hóa dấu cách.
Alias có giới hạn: person ↔ pedestrian, motorcycle ↔ motorbike, bicycle ↔ bike.
Car, truck, bus và các lớp khác cần label cùng tên hoặc mapping tường minh.
Nếu nhiều label cùng phù hợp, chương trình hiện danh sách tên/ID trong Terminal
và yêu cầu chọn đúng ID trước khi tải ảnh. Không tự chọn giữa Person/Pedestrian.
Nhập `skip` để bỏ qua lớp hoặc Enter để dừng. Sau khi chọn, Terminal in dòng
cấu hình có thể thêm dưới `[labels]` để dùng lại, ví dụ dạng `person = ID_thực`.
ID phải thuộc danh sách label rectangle/any của Job. Có thể dùng `person = ""`
để bỏ qua rõ ràng lớp person. Nếu tên đích trùng nhau sau chuẩn hóa, chọn bằng ID.
Chạy không có Terminal tương tác sẽ dừng và yêu cầu khai báo mapping trong config.
Pipeline detection chỉ dùng label `any`/`rectangle`. Pipeline scene dùng thêm label
`any`/`rectangle` cho các profile box. Label có attributes phải có default hợp lệ.

Lane detection, drivable area và semantic segmentation không được suy đoán từ
object detection. Scene dùng model segmentation riêng cho semantic; chương trình
vẫn liệt kê những label chưa có nguồn phù hợp. Audit reference không thay việc đối chiếu pixel khi chạy thật.
Weights mới phải là file cục bộ hoặc tên có trong danh sách assets của phiên bản
Ultralytics đã cài. Không âm thầm thay model nếu tải hoặc nạp weights thất bại.
Download chỉ tải weights, không gửi frame hoặc thông tin CVAT.

## Ghi annotation: chỉ sau khi kiểm tra dry-run

Dry-run 1, 5 frame và ghi thử 1 frame đã hoàn tất theo checkpoint hiện tại trên máy.
Để đi tiếp toàn Job dùng `--all --apply` ở trên. Lệnh ghi **1 frame** phía dưới dành
cho kiểm thử từng bước; không dùng nó nếu mục tiêu là xử lý tất cả ảnh.
Sau mỗi lần chạy, Terminal cũng in lệnh cho giai đoạn tiếp theo dựa trên checkpoint
cùng tài khoản, model, mapping và phạm vi. Không cần xóa checkpoint hay sửa source để bật ghi.

Xem kỹ danh sách đối tượng, mapping và frame ID. Lệnh ghi 1 frame:

```powershell
.\.venv\Scripts\python.exe .\main.py --organization cohort-5a --auth password --limit 1 --frame-timeout 180 --apply
```

Lệnh chạy lại suy luận, xem trước số box mới, xác minh schema của chính server,
rồi yêu cầu nhập chính xác **`APPLY JOB 206`** trong Terminal. Không có `--yes` để
bỏ qua xác nhận. Nếu bước ghi đầu tiên không có box mới thì không gửi PATCH và không
mở giai đoạn apply tiếp. Sau khi bước ghi 1 frame đã được xác minh, các bước sau
có thể không có box mới vì các đối tượng đã được gán trước đó. Khi ấy vẫn yêu cầu
xác nhận và kiểm tra lại phạm vi/metadata/labels/annotation, rồi ghi nhận giai đoạn
đã xử lý mà không gửi PATCH rỗng hoặc tạo box trùng.

Sau thông báo `Đã ghi và đọc lại xác minh ... rectangle`, mở
[Job 206](https://cvat.edufun.ai/tasks/71/jobs/206) và tải lại trang để xem nhãn.
Chỉ nhìn thấy dòng `DRY RUN hoàn tất` nghĩa là chưa ghi annotation.
Nếu `New rectangles prepared: 0`, không có box mới hợp lệ để gửi; xem lại số detected,
mapped, duplicates và mapping, không coi đó là lỗi đăng nhập.

Sau apply 1 frame thành công và đã xem nhãn trên CVAT, có thể dùng
`--limit 5 --apply`. Toàn Job với `--apply` chỉ mở sau dry-run toàn Job và apply 5 frame.
Mỗi lần đều yêu cầu xem trước và xác nhận riêng. Nếu thiếu quyền ghi, server có thể
trả 403 ở bước PATCH dù mọi GET đã thành công; chương trình dừng.

Chỉ gửi **`PATCH /api/jobs/206/annotations/?action=create`** với shapes mới,
không gửi ID annotation cũ, không có PUT/replace/delete. Trước ghi đọc lại Job,
labels, metadata và annotation, chống trùng lại rồi gửi tối đa 100 shapes mỗi request.
Sau mỗi request đọc lại để xác minh các ID mới và nội dung annotation cũ còn nguyên.
Job có consensus replicas chưa được hỗ trợ ghi.

Frame ID của CVAT là chỉ số theo Task, không đánh lại từ 0 trong Job. Metadata
`start_frame`/`included_frames` có thể theo dữ liệu gốc; chương trình chuyển qua
`step`, xử lý deleted frames, kiểm tra kích thước từng ảnh và chặn tọa độ ngoài ảnh.
Metadata không rõ ràng sẽ dừng thay vì đoán.

Chống trùng theo frame + label + IoU với rectangles và rectangle tracks có nội suy,
kể cả các box mới trong cùng lần chạy. Với rotated rectangles, cùng lớp/cùng frame
được bỏ qua thận trọng. Ngưỡng IoU không chứng minh hai dự đoán khác nhau là cùng
một đối tượng; cần xem lại kết quả. Không chạy đồng thời nhiều phiên ghi: server
không cung cấp giao dịch khóa xuyên suốt các lần đọc/ghi. Nếu timeout khi PATCH,
không tự gửi lại; đọc lại CVAT trước lần chạy tiếp. Nếu gián đoạn giữa các lô, các
lô đã xác nhận vẫn tồn tại; không tự rollback hoặc xóa annotation.

File `.annotation_state.json` chỉ lưu hash cấu hình/phạm vi/model/labels/metadata
và số frame hoàn thành để mở tuần tự 1 → 5 → toàn Job. Không chứa credentials,
ảnh hoặc annotation. Thay cấu hình/model/phạm vi sẽ cần thử lại 1 frame.
Checkpoint là điều kiện tuần tự cục bộ, không phải giấy phép ghi của server.

## Chẩn đoán API và HTTP 400/403

### Danh sách Labels, ẩn đối tượng và annotation thật

Ảnh chụp tab Labels chỉ cho thấy các loại nhãn có sẵn và trạng thái điều khiển.
Biểu tượng con mắt dùng để ẩn/hiện đối tượng của một label; khóa dùng để khóa chỉnh
sửa, không đồng nghĩa lỗi quyền API. Việc label có tên trong danh sách không chứng
minh đã có box/polygon/mask/track. Tài liệu: [CVAT Objects sidebar](https://docs.cvat.ai/docs/annotation/annotation-editor/objects-sidebar/).

Để xác minh nhãn nào thực sự có annotation trên toàn Job:

```powershell
.\.venv\Scripts\python.exe .\check_api.py --organization cohort-5a --auth password --inspect-annotations
```

Lệnh đọc Job, Organization, labels và annotations qua Job API, không đòi quyền đọc
Task cha, không tải ảnh, không nạp model hoặc ghi annotation. Báo cáo
`outputs/job_206_annotation_inventory.json` chỉ lưu tên/ID/type của labels và số
shapes/tracks/keyframes/tags theo từng loại geometry, không lưu tọa độ, ảnh hoặc bí mật.
Số đếm là toàn Job, không bảo đảm đối tượng đang nằm trên frame bạn đang xem.
Sau khi chạy có thể báo đã xong để Codex đọc file này và xác định cần gán thêm gì.

```powershell
.\.venv\Scripts\python.exe .\check_api.py --organization cohort-5a --auth password --diagnose-permissions
# Tên lệnh cũ cũng hoạt động:
.\.venv\Scripts\python.exe .\test_cvat.py --organization cohort-5a --auth password --diagnose-permissions
# Đối chiếu REST nếu cần:
.\.venv\Scripts\python.exe .\check_api.py --organization cohort-5a --backend rest --auth password --diagnose-permissions
# Chỉ kiểm tra endpoint công khai:
.\.venv\Scripts\python.exe .\check_api.py --public-only
```

`check_api.py`/`test_cvat.py` luôn chỉ đọc. Luồng chẩn đoán tổng thể vẫn báo exit code 1
khi Task bị từ chối; điều đó không phủ nhận các HTTP 200 của Job. Dùng `main.py`
cho luồng AI chỉ dựa vào quyền Job đã xác minh.

- **400 tại login**: request bị từ chối do validation. Đọc `detail`/`non_field_errors`
  đã lọc; chỉ mã status không đủ chứng minh sai password. Lỗi 400 trước đó chưa có
  phản hồi chi tiết nên không thể kết luận nguyên nhân lịch sử; hiện login đã 200.
- **401**: xác thực không được chấp nhận hoặc hết hạn.
- **403 tại Task sau `/users/self` 200**: xác thực thành công, quyền tài nguyên bị từ chối.
  Với tài khoản 212, Job assignee và membership đã được kiểm tra như mô tả phía trên.
- **403 tại Job/annotations/frame**: dừng; quản trị viên cần kiểm tra quyền tương ứng.
- **404**: tài nguyên sai ID, không tồn tại hoặc bị ẩn theo quyền.
- **429**: dừng, không lặp đăng nhập.
- **HTTP N/A**: có thể chưa nhận được HTTP (TLS/DNS/firewall) hoặc kiểm tra cục bộ
  thất bại. Xem tên bước và chi tiết đã lọc.

### Lỗi tải frame sau khi metadata đã thành công

`MaxRetryError` là lớp lỗi bao ngoài, chưa đủ để kết luận DNS, TLS hay timeout.
Công cụ phân loại chuỗi exception theo kiểu lỗi và in mã như `READ_TIMEOUT`,
`CONNECT_TIMEOUT`, `DNS`, `TLS_CERTIFICATE`, `CONNECTION_RESET`, `PROTOCOL`.
Nó không in URL, query string, headers, cookies hay thông báo exception gốc.

API metadata giữ timeout `--timeout` mặc định 30 giây. Riêng ảnh dùng thời gian
thiết lập kết nối tối đa 15 giây và read timeout `--frame-timeout` mặc định 120 giây
(giữa các lần nhận dữ liệu, không phải giới hạn tổng thời gian tải). Có thể dùng:

```powershell
.\.venv\Scripts\python.exe .\main.py --organization cohort-5a --auth password --limit 1 --frame-timeout 180
```

Nếu SDK gặp timeout, reset kết nối hoặc lỗi HTTP transport khi GET frame, chương
trình đối chiếu bằng `requests` đúng một lần với chính frame, Job, chất lượng gốc,
Organization và credentials đang giữ trong RAM. Không đăng nhập lại. Nếu thành
công, kết nối REST này được dùng cho các frame còn lại của lần chạy. Nếu REST cũng
thất bại, dừng và in loại lỗi để xử lý tiếp. Không fallback khi nhận HTTP 400/401/403,
lỗi chứng chỉ/TLS hoặc DNS. Luôn giữ kiểm tra SSL và chặn redirect.

Fallback này chỉ áp dụng **GET frame**; không tự thử lại login hoặc PATCH ghi nhãn.
Để đối chiếu riêng bằng REST từ đầu, thêm `--backend rest` vào lệnh dry-run phía trên.
Chưa có credentials trong phiên công cụ nên thay đổi tải ảnh được kiểm tra bằng
mock API; cần chạy lại tại Terminal để xác minh trên ảnh thật của trường.

Bí mật nhập bằng `getpass` hoặc biến môi trường; không có cờ nhận password/token
trên command line. `--auth pat` hỏi PAT ẩn và dùng Bearer nếu server hỗ trợ.
`--auth session` đăng nhập bằng API và giữ session trong RAM. `--auth auto` ưu tiên
`CVAT_ACCESS_TOKEN` nếu có, ngược lại hỏi username/password.
Không tắt SSL, không theo redirect, không tự đổi tài khoản/workspace để thử quyền.

`CVAT_URL`, `CVAT_ORGANIZATION`, `CVAT_TASK_ID`, `CVAT_JOB_ID`, `CVAT_USERNAME`,
`CVAT_PASSWORD`, `CVAT_ACCESS_TOKEN` được hỗ trợ. `.env.example` chỉ là mẫu;
chương trình không tự nạp `.env`. Dùng `--personal` để chọn rõ Personal Workspace
cho tài nguyên khác; Job 206 hiện dùng `cohort-5a`.

## Kiểm thử và các file

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q config.py auth.py cvat_client.py check_api.py test_cvat.py permission_diagnostics.py frame_mapping.py label_mapper.py label_coverage.py yolo_detector.py annotation_writer.py run_state.py network_diagnostics.py main.py tests
```

Unit test dùng dữ liệu giả và mock API, không truy cập hoặc ghi dữ liệu trường.
`config.py`/`annotation.toml`: cấu hình; `auth.py`: xác thực và lọc bí mật;
`cvat_client.py`: SDK/REST; `check_api.py`/`test_cvat.py`/`permission_diagnostics.py`:
chẩn đoán; `frame_mapping.py`: phạm vi và frame ID; `yolo_detector.py`: YOLO;
`label_mapper.py`: mapping; `annotation_writer.py`: bảo toàn và tạo nhãn;
`scene_detector.py`: kết hợp detection và semantic; `prepare_scene_model.py`: chuẩn bị weights semantic;
`run_state.py`: checkpoint; `network_diagnostics.py`: phân loại lỗi mạng đã bỏ chi tiết nhạy cảm;
`main.py`: chương trình chạy.
`.gitignore` loại trừ môi trường, bí mật, ảnh/video, weights, settings và checkpoint.

Tài liệu chính thức: [CVAT authentication](https://docs.cvat.ai/docs/api_sdk/sdk/lowlevel-api/),
[CVAT Job policy 2.76.0](https://github.com/cvat-ai/cvat/blob/v2.76.0/cvat/apps/engine/rules/jobs.rego),
[CVAT Job annotation implementation 2.76.0](https://github.com/cvat-ai/cvat/blob/v2.76.0/cvat/apps/dataset_manager/task.py),
[server schema](https://cvat.edufun.ai/api/schema/),
[Ultralytics prediction](https://docs.ultralytics.com/modes/predict/),
[urllib3 exceptions](https://urllib3.readthedocs.io/en/stable/reference/urllib3.exceptions.html),
[Requests timeouts](https://requests.readthedocs.io/en/latest/user/advanced/#timeouts).
