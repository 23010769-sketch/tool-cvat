# Kế hoạch hỗ trợ 31 labels của Job 206

**Định dạng mới theo yêu cầu: mọi nhãn là rectangle như car.** Profile scene
không còn xuất mask; pixel semantic được dùng nội bộ để bao box từng vùng rời.
Đổi các mask/đường tự động đã có bằng `main.py --convert-existing-to-boxes --all --apply`.
Lệnh này xem trước, cần nhập `CONVERT JOB 206 TO BOXES`, giữ ID/label/frame,
giữ các box cũ và nhãn vẽ tay; không chạy AI hay lấy ảnh. Thao tác cập nhật geometry
khác với luồng thêm nhãn thường và không chạy tự động khi chỉ chọn profile box.

`all_boxes_job206.toml` ánh xạ đủ 31 labels thành box. Với 9 lớp lane/drivable,
chỉ dùng nhãn tham chiếu nếu **mọi ảnh Job trùng tên và SHA-256 pixel RGB** với
ảnh BDD công khai. Không coi việc giống tên là đủ; ảnh khác pixel sẽ chặn ghi.
Không có weights lane mới và không tạo lane từ mask road. Biển báo dùng box nguồn
BDD khi đối chiếu đạt; các lớp detection/semantic khác chạy CPU cục bộ.

Nguồn tham chiếu: [BDD100K mirror công khai](https://huggingface.co/datasets/dronefreak/BDD100K/tree/69e9f9a60103df97e151e2606a6ad182db824f55/data/labels/100k),
revision `69e9f9a60103df97e151e2606a6ad182db824f55`. Đây là mirror cộng đồng,
không phải server của trường. JSON legacy giữ category lane/drivable và poly2d;
đường cong Bezier được lấy mẫu sai số tối đa 0.25 pixel trước khi lấy box.
Reference được đánh `source=file`, không gắn confidence AI; model đánh `source=auto`.
Tất cả ảnh chỉ ở RAM. Chỉ tải reference xuống để so sánh tại máy, không gửi ảnh CVAT ra ngoài.

Đã thử CPU trên ảnh công khai `b1c66a42-6f7d68ca.jpg`: 31 mapping, 66 rectangle,
trong đó 15 lane và 6 traffic sign. Chưa xác minh rằng 20 ảnh Job 206 là các ảnh
BDD tương ứng; bước này bắt buộc trong Terminal đăng nhập của người dùng.

Đã đọc kiểm kê API trong `outputs/job_206_annotation_inventory.json` do người dùng
chạy bằng tài khoản của mình. Server `https://cvat.edufun.ai`, Organization `cohort-5a`,
Task 71, Job 206. **Cả 31 labels có type=any.** Báo cáo có 97 rectangle:
pedestrian (ID 637): 8; car (639): 83; truck (640): 3; bus (641): 3.
27 labels còn lại chưa có shapes/tracks/tags. Đây là thông tin trong báo cáo,
không phải số đếm từ một phiên đăng nhập mới của công cụ.

Không lấy việc giống tên làm bằng chứng hai label có cùng mục đích. Profile mới
chỉ rõ label ID và geometry; mỗi lần chạy vẫn kiểm tra lại labels và quyền.

| Nhóm | Labels | Trạng thái triển khai |
| --- | --- | --- |
| Detection, 10 nhãn | pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign | Có profile `bdd_detection.toml`; weights BDD100K YOLO11n đã tải, kiểm tra checksum/lớp và thử CPU bằng ảnh trống. Chưa chạy profile mới trên ảnh trường. |
| Drivable, 2 nhãn | area/alternative, area/drivable | Box từ reference BDD nếu ảnh trùng pixel; chưa có weights riêng. Không thay bằng mask road. |
| Lane, 7 nhãn | lane/crosswalk, lane/double white, lane/double yellow, lane/road curb, lane/single other, lane/single white, lane/single yellow | Box từ từng lane đúng category trong reference BDD, chỉ khi ảnh trùng pixel; chưa có weights phân loại lane riêng. |
| Semantic, 12 nhãn | road, sidewalk, building, wall, fence, pole, traffic_light, traffic_sign, vegetation, terrain, sky, person | SegFormer-B0 Cityscapes + box từng vùng rời trong `scene_job206.toml`, ID 1750–1761. |

## Detection đã chuẩn bị

Nguồn: [model card của tác giả](https://huggingface.co/dronefreak/bdd100k-yolo11n).
Revision `2b5554133522f96f50ce87a4df0a2de556e4b7ad`, file `best.pt`.
SHA-256 `c120d075bcc6aaf94523f9a2c9d681819ebd692f36eb5ea64656d1c5f1278a70`.
File cục bộ `models/bdd100k-yolo11n.pt`.

Model khai báo lớp `person, rider, car, truck, bus, train, motor, bike, traffic light, traffic sign`.
Profile ánh xạ ID 637–646: `person → pedestrian`, `motor → motorcycle`, `bike → bicycle`.
Không tự nhân một box sang `person` semantic hoặc cả hai cặp tên traffic light/sign.
Đã điền đúng ID của nhãn detection để không hỏi lại giữa traffic light/traffic_light.

Đây là model cộng đồng đã huấn luyện cho BDD100K, không phải model riêng của trường.
Model card báo kết quả yếu ở lớp hiếm, đặc biệt train; có tên lớp không bảo đảm
phát hiện được mọi đối tượng. Cần dry-run và kiểm tra trên ảnh thực tế trước khi ghi.

## Pipeline kết hợp 22/31 labels

`scene_job206.toml` kết hợp model BDD detection và SegFormer-B0 Cityscapes:
10 lớp detection tạo rectangle vào ID 637–646; 12 lớp semantic tạo rectangle vào ID
1750–1761. Không dùng mask `road` để thay `area/drivable`, không tạo lane giả.
Hai vùng drivable và bảy loại lane còn thiếu weights/backend phù hợp đã xác minh.

Đã cài `transformers==4.57.6` trong `.venv`. Nếu thiết lập lại môi trường:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-scene.txt
.\.venv\Scripts\python.exe .\prepare_bdd_model.py
.\.venv\Scripts\python.exe .\prepare_scene_model.py
```

Chạy một phiên xử lý theo các giai đoạn thử và ghi:

```powershell
.\.venv\Scripts\python.exe .\main.py --organization cohort-5a --auth password --config .\scene_job206.toml --all --apply --frame-timeout 180
```

Checkpoint pipeline mới tách riêng do thay weights/mapping/geometry. Chương trình
tự chạy các bước dry-run còn thiếu; trước mỗi bước ghi cần nhập `APPLY JOB 206`.
Kiểm tra box trên CVAT sau bước ghi 1 ảnh rồi mới xác nhận bước lớn hơn.
Lệnh này chỉ gán **22 labels đã hỗ trợ**, không bảo đảm mỗi lớp xuất hiện trong mọi ảnh.
Nếu thêm `--require-all-labels`, chương trình dừng do 9 labels còn thiếu.

Chế độ native cũ (cấu hình riêng `output_geometry="native"`) vẫn dùng mask RLE chính thức của SDK. Khi đã có annotation
semantic, chỉ thêm pixel chưa được các semantic labels đã cấu hình chiếm giữ;
không sửa/xóa mask hoặc polygon cũ. Semantic track hoặc geometry không thể
rasterize an toàn sẽ chặn ghi. Box detection cũ không bị đổi thành mask.
Box mới được chống trùng với ngưỡng IoU 0.70; vẫn cần xem lại chất lượng khi đổi model.

SegFormer revision `21b3847fae21ddee674abd31129307b6a1235bd9`, weights SHA-256
`027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c`.
Chỉ nạp weights cục bộ đã kiểm tra, không gửi ảnh lên Hugging Face hoặc nơi khác.

## Audit thêm và các nhóm còn lại

Đọc ID/type thực tế mà không tải ảnh hoặc ghi annotation:

```powershell
.\.venv\Scripts\python.exe .\main.py --organization cohort-5a --auth password --config .\bdd_detection.toml --audit-labels
```

Báo cáo được lưu tại `outputs/job_206_label_audit.json`. File chỉ có thông tin labels
và mức hỗ trợ, không chứa ảnh, password, token hoặc cookie; `outputs/` đã được gitignore.
Sau khi chạy, có thể báo đã xong để Codex đọc file này, không cần dán mật khẩu hoặc ảnh.

Đổi model/profile làm checkpoint riêng theo hash weights/mapping; không xóa checkpoint cũ.
Nếu muốn thử detection mới, bắt đầu `--config .\bdd_detection.toml --limit 1` ở DRY RUN.
`--require-all-labels` sẽ chặn ghi nếu các nhóm còn thiếu chưa được hỗ trợ.

Nguồn đối chiếu:

- [Taxonomy BDD100K chính thức](https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst).
- [Config SegFormer-B0 Cityscapes](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024/blob/main/config.json).
- [BDD100K model zoo](https://github.com/SysCV/bdd100k-models): có semantic/drivable; lane marking còn nằm trong roadmap.
- [UNet lane đã kiểm tra](https://github.com/nick8592/lane-detection-unet-ncnn/blob/main/config/test_config.yaml) có `out_channels: 1`, nên không đủ phân biệt bảy loại lane trên.
