"""BDD reference labels, only after exact local RGB comparison of every Job frame.

This is reference transfer, not lane-model inference. Public downloads use a
separate unauthenticated session. CVAT pixels never leave the machine or RAM.
"""

import hashlib
import io
import json
import math
import re
from bisect import bisect_right
from collections import Counter

import requests
from PIL import Image
from tqdm import tqdm

from yolo_detector import Detection


REPO = "dronefreak/BDD100K"
REVISION = "69e9f9a60103df97e151e2606a6ad182db824f55"
CONVERTER_VERSION = "lane-reference-v1"
LANES = ("lane/crosswalk", "lane/double white", "lane/double yellow", "lane/road curb",
         "lane/single other", "lane/single white", "lane/single yellow")
NAMES = dict(enumerate(("traffic sign", *LANES)))
BOX_NAMES = {**NAMES, 8: "area/alternative", 9: "area/drivable"}
FORMATS = {key: "rectangle" if name == "traffic sign" else "polyline" for key, name in NAMES.items()}
BDD_NAME = re.compile(r"[0-9a-f]{8}-[0-9a-f]{8}")


def frame_names(job, metadata, frames):
    """Use the same dense metadata index as build_frames, including sparse Jobs."""
    if job.get("mode") != "annotation":
        raise ValueError("BDD reference chỉ hỗ trợ Job ảnh độc lập.")
    media = metadata.get("frames", [])
    result = {}
    for number in frames:
        name = media[number - job["start_frame"]].get("name", "").replace("\\", "/").rsplit("/", 1)[-1]
        if not name.endswith(".jpg") or not BDD_NAME.fullmatch(name[:-4]):
            raise ValueError(f"Frame {number} không giữ tên ảnh BDD100K gốc. Chưa xác định được nguồn lane; không ghi nhãn.")
        result[number] = name[:-4]
    if len(set(result.values())) != len(result):
        raise ValueError("Job có tên ảnh BDD trùng; cần xác minh thủ công trước khi dùng reference.")
    return result


class PublicReferenceStore:
    def __init__(self):
        self.session = requests.Session()  # Never reuse the CVAT session/headers/cookies.
        self.session.headers.update({"Accept": "application/json, image/jpeg", "User-Agent": "cvat-local-bdd-reference/1"})
        self.boundaries = {}
        self.records = {}

    def close(self):
        self.records.clear()
        self.session.close()

    def get(self, path, *, api=False, limit=2_000_000, missing=False):
        base = (f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/" if api else
                f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/")
        try:
            with self.session.get(base + path, params={"limit": 1} if api else None,
                                  timeout=(15, 60), stream=True) as response:
                if response.status_code == 404 and missing:
                    return None
                if response.status_code != 200:
                    raise ValueError(f"Nguồn BDD công khai trả HTTP {response.status_code}; dừng, không ghi.")
                raw = bytearray()
                for block in response.iter_content(65536):
                    raw.extend(block)
                    if len(raw) > limit:
                        raise ValueError("File BDD vượt giới hạn kích thước; dừng.")
                return bytes(raw)
        except requests.RequestException:
            raise ValueError("Không tải được reference BDD qua HTTPS; kiểm tra kết nối, không tắt SSL.") from None

    def _boundaries(self, split):
        if split not in self.boundaries:
            # Fixed shards verified at the pinned revision. Only public file lists.
            values = []
            for number in range(9 if split == "train" else 2):
                shard = f"data/labels/100k/{split}/shard_{number:03d}"
                first = json.loads(self.get(shard, api=True))
                if (not isinstance(first, list) or len(first) != 1 or first[0].get("type") != "file"
                        or not first[0].get("path", "").startswith(shard + "/")):
                    raise ValueError("Cấu trúc shard BDD khác revision đã xác minh.")
                name = first[0]["path"].rsplit("/", 1)[-1]
                if not name.endswith(".json") or not BDD_NAME.fullmatch(name[:-5]):
                    raise ValueError("Tên file trong shard BDD không hợp lệ.")
                values.append((name[:-5], number))
            self.boundaries[split] = sorted(values)
        return self.boundaries[split]

    def record(self, stem):
        if not BDD_NAME.fullmatch(stem):
            raise ValueError("Tên ảnh không phải ID BDD100K.")
        if stem in self.records:
            return self.records[stem]
        for split in ("val", "train"):
            boundaries = self._boundaries(split)
            index = bisect_right([row[0] for row in boundaries], stem) - 1
            if index < 0:
                continue
            shard = f"{split}/shard_{boundaries[index][1]:03d}"
            raw = self.get(f"data/labels/100k/{shard}/{stem}.json", missing=True)
            if raw is None:
                continue
            data = json.loads(raw)
            if (not isinstance(data, dict) or data.get("name") != stem
                    or len(data.get("frames", [])) != 1
                    or not isinstance(data["frames"][0].get("objects"), list)):
                raise ValueError("Annotation nguồn BDD không khớp tên ảnh/cấu trúc đã xác minh.")
            value = (data, f"data/images_100k/100k/{shard}/{stem}.jpg", hashlib.sha256(raw).hexdigest())
            self.records[stem] = value
            return value
        raise ValueError("Không tìm thấy nhãn BDD train/val cho ảnh này; chưa có lane model thay thế, không ghi.")


def flatten_curve(poly):
    """Legacy BDD uses L vertices and triples of cubic C control/end vertices."""
    if not isinstance(poly, list) or not 2 <= len(poly) <= 4096:
        raise ValueError("poly2d BDD không hợp lệ.")
    if any(not isinstance(row, list) or len(row) != 3
           or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 100000 for v in row[:2])
           or row[2] not in ("L", "C") for row in poly) or poly[0][2] != "L":
        raise ValueError("Không hiểu mã đường cong BDD; không đoán geometry.")
    result = [tuple(poly[0][:2])]

    def midpoint(a, b):
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

    def distance(point, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        denom = dx * dx + dy * dy
        t = min(1., max(0., ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denom)) if denom else 0.
        return math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy)

    def cubic(a, b, c, d, depth=0):
        if max(distance(b, a, d), distance(c, a, d)) <= .25:
            result.append(d)
        else:
            if depth >= 16 or len(result) >= 4096:
                raise ValueError("Đường cong BDD quá phức tạp; dừng.")
            ab, bc, cd = midpoint(a, b), midpoint(b, c), midpoint(c, d)
            abc, bcd = midpoint(ab, bc), midpoint(bc, cd)
            center = midpoint(abc, bcd)
            cubic(a, ab, abc, center, depth + 1)
            cubic(center, bcd, cd, d, depth + 1)

    index = 1
    while index < len(poly):
        if poly[index][2] == "L":
            result.append(tuple(poly[index][:2]))
            index += 1
        else:
            if index + 2 >= len(poly) or any(row[2] != "C" for row in poly[index:index + 3]):
                raise ValueError("Bezier BDD thiếu bộ ba C; dừng.")
            cubic(result[-1], *(tuple(row[:2]) for row in poly[index:index + 3]))
            index += 3
    if len(result) > 4096:
        raise ValueError("Polyline BDD vượt giới hạn điểm.")
    return result


def clip_polyline(vertices, width, height):
    """Liang–Barsky clipping splits paths outside the image, never joins gaps."""
    parts, current = [], []
    for a, b in zip(vertices[:-1], vertices[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        low, high = 0., 1.
        valid = True
        for p, q in ((-dx, a[0]), (dx, width - a[0]), (-dy, a[1]), (dy, height - a[1])):
            if p == 0:
                if q < 0:
                    valid = False
                    break
            elif p < 0:
                low = max(low, q / p)
            else:
                high = min(high, q / p)
        if low > high:
            valid = False
        if not valid:
            if len(current) >= 2:
                parts.append(current)
            current = []
            continue
        left, right = [(min(float(width), max(0., a[0] + t * dx)),
                        min(float(height), max(0., a[1] + t * dy))) for t in (low, high)]
        if left == right:
            continue
        if current and math.dist(current[-1], left) > 1e-6:
            if len(current) >= 2:
                parts.append(current)
            current = []
        if not current:
            current.append(left)
        current.append(right)
        if high < 1:
            parts.append(current)
            current = []
    if len(current) >= 2:
        parts.append(current)
    return parts


def enclosing_box(vertices, width, height):
    x1, y1 = max(0., min(p[0] for p in vertices)), max(0., min(p[1] for p in vertices))
    x2, y2 = min(float(width), max(p[0] for p in vertices)), min(float(height), max(p[1] for p in vertices))
    if x1 > x2 or y1 > y2:
        raise ValueError("Geometry hoàn toàn ngoài ảnh.")
    # A perfectly vertical/horizontal lane still needs a positive-area box.
    if x1 == x2:
        x1, x2 = max(0., x1 - .5), min(float(width), x2 + .5)
    if y1 == y2:
        y1, y2 = max(0., y1 - .5), min(float(height), y2 + .5)
    return tuple(round(v, 6) for v in (x1, y1, x2, y2))


def convert_objects(data, width, height, *, boxes=False):
    classes = {name: key for key, name in (BOX_NAMES if boxes else NAMES).items()}
    result, ignored = [], Counter()
    for obj in data["frames"][0]["objects"]:
        name = obj.get("category")
        if name not in classes:
            if isinstance(name, str) and name.startswith("lane/"):
                ignored[name] += 1
            continue
        if name == "traffic sign":
            box = obj.get("box2d", {})
            points = tuple(box.get(key) for key in ("x1", "y1", "x2", "y2"))
            if (any(type(p) not in (float, int) or not math.isfinite(p) for p in points)
                    or not 0 <= points[0] < points[2] <= width + 1
                    or not 0 <= points[1] < points[3] <= height + 1):
                raise ValueError("Box biển báo BDD không hợp lệ.")
            result.append(Detection(classes[name], 1., points, "rectangle", "file"))
        else:
            vertices = flatten_curve(obj.get("poly2d"))
            if boxes:
                result.append(Detection(classes[name], 1., enclosing_box(vertices, width, height), "rectangle", "file"))
                continue
            parts = clip_polyline(vertices, width, height)
            if not parts:
                raise ValueError("Lane nguồn hoàn toàn ngoài ảnh; dừng để kiểm tra.")
            for part in parts:
                points = tuple(round(p, 6) for pair in part for p in pair)
                result.append(Detection(classes[name], 1., points, "polyline", "file"))
    return result, ignored


class BddReferenceDetector:
    names = NAMES
    shape_types = FORMATS

    def __init__(self, settings):
        self.settings = settings
        self.boxes = settings.output_geometry == "rectangle"
        if self.boxes:
            self.names = BOX_NAMES
            self.shape_types = {key: "rectangle" for key in self.names}
        self.weights_hash = f"{REPO}@{REVISION}:{CONVERTER_VERSION}"
        self.verified = {}

    def prepare(self, client, job, metadata, scope):
        names = frame_names(job, metadata, scope.frames)  # All names before any public request.
        self.verified.clear()
        provenance = {}
        store = PublicReferenceStore()
        print("[REFERENCE BDD] Dùng annotation nguồn, không chạy AI lane. Source CVAT=file.")
        print("Đối chiếu từng pixel RGB của TẤT CẢ frame trước mọi bước ghi. Ảnh chỉ ở RAM.")
        try:
            for number in tqdm(sorted(scope.frames), desc="Verify BDD frames", unit="frame"):
                data, image_path, digest = store.record(names[number])
                public_raw = store.get(image_path, limit=5_000_000)
                info = scope.frames[number]
                with Image.open(io.BytesIO(public_raw)) as public:
                    if public.size != (info.width, info.height):
                        raise ValueError(f"Frame {number}: kích thước khác ảnh BDD gốc; không ghi.")
                    public_rgb = public.convert("RGB")
                    try:
                        public_hash = hashlib.sha256(public_rgb.tobytes()).digest()
                    finally:
                        public_rgb.close()
                del public_raw
                cvat_raw = client.frame_bytes(number, scope.frames)
                with Image.open(io.BytesIO(cvat_raw)) as actual:
                    if actual.size != (info.width, info.height):
                        raise ValueError(f"Frame {number}: kích thước khác metadata; không ghi.")
                    actual_rgb = actual.convert("RGB")
                    try:
                        actual_hash = hashlib.sha256(actual_rgb.tobytes()).digest()
                    finally:
                        actual_rgb.close()
                del cvat_raw
                if actual_hash != public_hash:
                    raise ValueError(f"Frame {number}: pixel KHÔNG trùng ảnh BDD gốc; không chuyển nhãn, không ghi.")
                detections, ignored = convert_objects(data, info.width, info.height, boxes=self.boxes)
                self.verified[number] = detections
                provenance[number] = {"json_sha256": digest, "rgb_sha256": actual_hash.hex()}
                counts = Counter(self.names[d.class_id] for d in detections)
                tqdm.write(f"[TRÙNG KHỚP] frame={number} | " + "; ".join(f"{k}: {v}" for k, v in counts.items()))
                if ignored:
                    tqdm.write("[BỎ QUA] Lane không có label đích: " + ", ".join(ignored))
            self.weights_hash += ":" + hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        except Exception:
            self.verified.clear()
            raise
        finally:
            store.close()
        print(f"[OK] {len(self.verified)}/{len(scope.frames)} ảnh trùng chính xác với BDD. Chưa ghi annotation.")

    def for_frames(self, numbers):
        if any(number not in self.verified for number in numbers):
            raise ValueError("Frame chưa được đối chiếu pixel; không được chuyển nhãn.")
        return [self.verified[number] for number in numbers]


class BoxSceneDetector:
    """31 label mappings: local detection/semantic boxes plus verified BDD boxes."""

    def __init__(self, settings):
        from scene_detector import SceneDetector
        self.scene = SceneDetector(settings)
        self.reference = BddReferenceDetector(settings)
        self.names = {**self.scene.names,
                      **{200 + key: "reference/" + name for key, name in self.reference.names.items() if key != 0}}
        self.shape_types = {key: "rectangle" for key in self.names}
        self.weights_hash = self.scene.weights_hash + ":" + self.reference.weights_hash

    def prepare(self, client, job, metadata, scope):
        self.reference.prepare(client, job, metadata, scope)
        self.weights_hash = self.scene.weights_hash + ":" + self.reference.weights_hash

    def predict_for_frames(self, images, numbers):
        signs = [key for key, name in self.scene.names.items() if name == "det/traffic sign"]
        if len(signs) != 1:
            raise ValueError("Không xác minh được lớp biển báo của model detection.")
        results = self.scene.predict(images)
        references = self.reference.for_frames(numbers)
        for index, (predictions, source) in enumerate(zip(results, references)):
            # Prefer exact reference sign boxes over adding YOLO duplicates.
            predictions = [d for d in predictions if d.class_id != signs[0]]
            for item in source:
                target = signs[0] if item.class_id == 0 else 200 + item.class_id
                predictions.append(Detection(target, item.confidence, item.points, "rectangle", "file"))
            results[index] = predictions
        return results
