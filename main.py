"""AI annotation qua Job API; mặc định DRY RUN, không ghi ảnh xuống đĩa."""

import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from annotation_writer import apply_shapes, deduplicate, make_shape, verify_create_schema
from auth import DiagnosticError, Redactor, explain_status, read_credentials
from config import Config, DetectionConfig, load_detection_config, parse_config
from cvat_client import CvatClient
from frame_mapping import JobScope, build_frames, validate_job
from label_mapper import AmbiguousMappingError, LabelMapper
from label_coverage import label_coverage, print_label_coverage, require_full_coverage, save_label_audit
from run_state import RunState
from yolo_detector import YoloDetector


def load_detector(settings):
    if settings.pipeline == "scene-reference":
        from bdd_reference import BoxSceneDetector
        return BoxSceneDetector(settings)
    if settings.pipeline == "bdd-reference":
        from bdd_reference import BddReferenceDetector
        return BddReferenceDetector(settings)
    if settings.pipeline == "scene":
        from scene_detector import SceneDetector
        return SceneDetector(settings)
    return YoloDetector(settings)


def detector_formats(detector):
    formats = getattr(detector, "shape_types", None)
    return formats if isinstance(formats, dict) else {key: "rectangle" for key in detector.names}


def print_next_action(state, config, args):
    action = state.next_action()
    if action is None:
        print("[OK] Đã hoàn tất các giai đoạn thử và ghi. Mở CVAT để kiểm tra chất lượng nhãn.")
        return
    limit, apply = action
    scope = f"{limit} frame" if limit is not None else "toàn Job"
    print(f"[TIẾP THEO] {'Ghi annotation thật' if apply else 'DRY RUN'}: {scope}.")
    parts = [r".\.venv\Scripts\python.exe", r".\main.py"]

    def option(flag, value):
        # Printed commands use PowerShell literal quoting, never credentials.
        parts.extend((flag, "'" + str(value).replace("'", "''") + "'"))

    if config.organization:
        option("--organization", config.organization)
    else:
        parts.append("--personal")
    if config.url != "https://cvat.edufun.ai":
        option("--url", config.url)
    if config.task_id != 71:
        option("--task-id", config.task_id)
    if config.job_id != 206:
        option("--job-id", config.job_id)
    option("--auth", args.auth)
    option("--backend", config.backend)
    if config.timeout != 30:
        option("--timeout", f"{config.timeout:g}")
    option("--frame-timeout", f"{config.frame_timeout:g}")
    if Path(args.config).resolve() != Path(__file__).with_name("annotation.toml").resolve():
        option("--config", args.config)
    if limit is not None:
        option("--limit", limit)
    if apply:
        parts.append("--apply")
    if getattr(args, "require_all_labels", False):
        parts.append("--require-all-labels")
    print(" ".join(parts))
    if apply:
        print(f"Lệnh sẽ xem trước và yêu cầu nhập APPLY JOB {config.job_id} trước khi gửi PATCH.")


def resolve_label_mapping(names, labels, overrides, redactor, *, shape_types=None):
    while True:
        try:
            return LabelMapper(names, labels, overrides, shape_types=shape_types)
        except AmbiguousMappingError as error:
            print(f"[CẦN CHỌN NHÃN] YOLO '{redactor.clean(error.source)}' khớp các label CVAT:")
            choices = {label["id"] for label in error.matches}
            for label in error.matches:
                print(f"  ID {label['id']} | {redactor.clean(label['name'])} | type={redactor.clean(label.get('type', 'any'))}")
            if not sys.stdin.isatty():
                raise ValueError("Cần chọn label trong Terminal hoặc điền label ID vào [labels] của annotation.toml.") from None
            while True:
                answer = input("Chọn label ID đúng nghĩa; 'skip' để bỏ qua lớp này; Enter để dừng: ").strip()
                if not answer:
                    raise ValueError("Chưa chọn label; đã dừng trước khi tải ảnh.") from None
                if answer.casefold() == "skip":
                    value = ""
                    break
                if answer.isascii() and answer.isdecimal() and int(answer) in choices:
                    value = int(answer)
                    break
                print("ID không thuộc danh sách vừa hiển thị; hãy chọn lại.")
            overrides[error.source] = value
            key = json.dumps(error.source, ensure_ascii=False)
            print(f"Để dùng lại lựa chọn, thêm dòng này dưới [labels] trong annotation.toml: {key} = {json.dumps(value)}")


def process_frames(client, scope, detector, mapper, settings, limit):
    numbers = sorted(scope.frames)
    if limit is not None:
        numbers = numbers[:limit]
    candidates = []
    counts = dict(processed=0, detected=0, mapped=0, unmapped=0, invalid=0)
    with tqdm(total=len(numbers), desc="Job frames", unit="frame") as progress:
        for start in range(0, len(numbers), settings.batch_size):
            batch_numbers = numbers[start:start + settings.batch_size]
            images = []
            try:
                reference = settings.pipeline == "bdd-reference"
                for number in ([] if reference else batch_numbers):
                    raw = client.frame_bytes(number, scope.frames)
                    with io.BytesIO(raw) as buffer, Image.open(buffer) as decoded:
                        info = scope.frames[number]
                        if decoded.size != (info.width, info.height):
                            raise ValueError(f"Kích thước ảnh frame {number} khác metadata; dừng.")
                        decoded.load()
                        images.append(decoded.convert("RGB"))
                    del raw
                results = (detector.for_frames(batch_numbers) if reference else
                           detector.predict_for_frames(images, batch_numbers) if settings.pipeline == "scene-reference" else
                           detector.predict(images))
                for number, detections in zip(batch_numbers, results):
                    for detection in detections:
                        counts["detected"] += 1
                        label = mapper.label(detection.class_id)
                        name = client.redactor.clean(detector.names.get(detection.class_id, "unknown"))
                        if label is None:
                            counts["unmapped"] += 1
                            tqdm.write(f"frame={number} | {name} | conf={detection.confidence:.3f} | bỏ qua: chưa có mapping")
                            continue
                        try:
                            shape = make_shape(number, detection, label, mapper, scope)
                        except ValueError:
                            counts["invalid"] += 1
                            tqdm.write(f"frame={number} | bỏ qua: box hoặc thuộc tính không hợp lệ")
                            continue
                        counts["mapped"] += 1
                        candidates.append(shape)
                        details = (f"xyxy={shape['points']}" if shape["type"] == "rectangle" else
                                   f"polyline_vertices={len(shape['points']) // 2}" if shape["type"] == "polyline" else
                                   f"mask_bbox={shape['points'][-4:]}; runs={len(shape['points']) - 4}")
                        origin = "source=file; reference BDD" if shape["source"] == "file" else f"conf={detection.confidence:.3f}"
                        tqdm.write(f"frame={number} | {name} -> label_id={label['id']} | "
                                   f"{origin} | {details}")
                    counts["processed"] += 1
                    progress.update(1)
            finally:
                for image in images:
                    image.close()
    return candidates, counts


@dataclass
class StageResult:
    code: int = 0
    completed: bool = False


@dataclass
class AnnotationSession:
    client: CvatClient
    config: Config
    scope: JobScope
    detector: YoloDetector
    mapper: LabelMapper
    settings: DetectionConfig
    state: RunState
    job: dict
    existing: dict | None = None

    def run_stage(self, limit, apply):
        self.state.require(limit, apply)
        existing = self.existing if self.existing is not None else self.client.annotations()
        self.existing = None  # Refresh before each later stage in the same session.
        frame_count = min(limit, len(self.scope.frames)) if limit is not None else len(self.scope.frames)
        print(f"[GIAI ĐOẠN] {'APPLY' if apply else 'DRY RUN'} — {frame_count}/{len(self.scope.frames)} frame hợp lệ trong Job {self.config.job_id}")
        print("[4/5] Processing frames...")
        candidates, counts = process_frames(self.client, self.scope, self.detector, self.mapper, self.settings, limit)
        print("[5/5] Preparing annotation results...")
        pending, duplicate_count = deduplicate(candidates, existing, self.settings.duplicate_iou, scope=self.scope)
        noun = "rectangle" if self.scope.allowed_shape_types == {"rectangle"} else "shape"
        print(f"Processed frames: {counts['processed']}\nDetected objects: {counts['detected']}")
        print(f"Mapped objects: {counts['mapped']}\nSkipped objects: {counts['unmapped'] + counts['invalid'] + duplicate_count}")
        print(f"Unmapped: {counts['unmapped']}; invalid: {counts['invalid']}; duplicates: {duplicate_count}")
        print(f"New {noun}s prepared: {len(pending)}")
        if not apply:
            self.state.record(limit, False, counts["processed"])
            print("[OK] DRY RUN hoàn tất; không ghi annotation. Checkpoint chỉ lưu hash và số frame.")
            return StageResult(completed=True)
        if self.job.get("parent_job_id") or self.job.get("replicas_count", 0):
            raise ValueError("Chưa hỗ trợ ghi Job có consensus replicas; dry-run vẫn dùng được.")
        schema_verified = verify_create_schema(self.client.schema(), required_types=self.scope.allowed_shape_types,
                                               required_sources=self.scope.allowed_sources if "file" in self.scope.allowed_sources else None)
        print("[OK] Schema server xác nhận PATCH Job/action=create. Quyền ghi sẽ do server quyết định khi gửi.")
        if not pending:
            print("Không có shape mới để ghi; không gửi PATCH rỗng.")
            if limit == 1 or not self.state.completed.get("apply_1"):
                print("Chưa kiểm chứng được bước ghi 1 frame; không tự mở rộng giai đoạn.")
                return StageResult()
            print("Bước ghi 1 frame đã được xác minh trước đó. Có thể xác nhận kiểm tra giai đoạn này để tiếp tục, không tạo nhãn trùng.")
        if not sys.stdin.isatty():
            raise ValueError("--apply cần Terminal tương tác để xác nhận.")
        phrase = f"APPLY JOB {self.config.job_id}"
        print(f"Giai đoạn {frame_count} frame: sẽ thêm tối đa {len(pending)} {noun} vào Job {self.config.job_id}; giữ annotation cũ.")
        if input(f"Nhập chính xác '{phrase}' để xác nhận: ").strip() != phrase:
            print("Đã hủy; dừng toàn bộ các giai đoạn còn lại.")
            return StageResult(code=2)
        created, extra_duplicates = apply_shapes(self.client, pending, self.scope, self.settings,
                                                  confirmed=True, schema_verified=schema_verified)
        print(f"Created: {created}; duplicates skipped after refresh: {extra_duplicates}")
        # A later stage may need no new boxes because the objects already exist.
        # apply_shapes still revalidates scope/metadata/labels/annotations above.
        completed = bool(created) or limit != 1 and bool(self.state.completed.get("apply_1"))
        if completed:
            self.state.record(limit, True, counts["processed"])
            print(f"[OK] Đã xử lý và xác minh {counts['processed']} frame; thêm {created} {noun} trong Job {self.config.job_id}.")
            print(f"Mở CVAT: {self.config.url.rstrip('/')}/tasks/{self.config.task_id}/jobs/{self.config.job_id}")
            print("Tải lại trang Job và kiểm tra độ chính xác trước khi xác nhận giai đoạn tiếp theo.")
        if created:
            print(f"[OK] Đã ghi và đọc lại xác minh {created} {noun} trong Job {self.config.job_id}.")
        return StageResult(completed=completed)

    def run_all(self, apply):
        print(f"[TOÀN JOB] {len(self.scope.frames)} frame hợp lệ; chỉ xử lý Job {self.config.job_id}.")
        while True:
            action = self.state.next_action() if apply else self.state.next_dry_action()
            if action is None:
                print(f"[OK] Đã hoàn tất {'luồng gán nhãn' if apply else 'dry-run'} toàn Job: {len(self.scope.frames)} frame hợp lệ.")
                print("Một ảnh có thể không có annotation mới nếu model không phát hiện đối tượng/vùng được ánh xạ hoặc nhãn đã tồn tại.")
                print(f"Mở CVAT: {self.config.url.rstrip('/')}/tasks/{self.config.task_id}/jobs/{self.config.job_id}")
                return 0
            result = self.run_stage(*action)
            if result.code:
                return result.code
            if not result.completed:
                print("[DỪNG] Chưa hoàn tất giai đoạn hiện tại; không bỏ qua kiểm tra để ghi các ảnh còn lại.")
                return 1


def main(argv=None):
    config, args = parse_config(argv, annotation=True)
    redactor, client = Redactor(), None
    print("=" * 45)
    print("CVAT YOLO AUTO ANNOTATION")
    print(f"Server: {config.url}\nOrganization: {config.organization or 'Personal Workspace'}")
    print(f"Task ID: {config.task_id}\nJob ID: {config.job_id}")
    print("Mode:", "CONVERT TO BOXES — APPLY" if args.convert_existing_to_boxes and args.apply else
          "CONVERT TO BOXES — DRY RUN" if args.convert_existing_to_boxes else
          "ANNOTATION INVENTORY" if args.inspect_annotations else "LABEL AUDIT" if args.audit_labels
          else "APPLY — cần xem trước và xác nhận" if args.apply else "DRY RUN")
    try:
        settings, overrides = load_detection_config(args.config)
        client = CvatClient(config, redactor)
        if client.about().get("version") != "2.76.0":
            raise ValueError("Server khác 2.76.0; cần kiểm tra lại schema/API.")
        if args.public_only:
            print("Endpoint công khai hoạt động; chưa kiểm tra xác thực.")
            return 0
        print("[1/5] Checking authentication...")
        client.login(read_credentials(config.auth, args.username, redactor))
        user = client.current_user()
        client.set_organization()
        if args.inspect_annotations:
            from annotation_inventory import inspect_annotations
            inspect_annotations(client, config, redactor)
            return 0
        if args.diagnose_permissions:
            from permission_diagnostics import diagnose_permissions
            return diagnose_permissions(client, config, user)
        print("[2/5] Checking Task and Job scope via Job API...")
        job = client.job()
        organization = client.organization(job["organization"]) if config.organization else None
        validate_job(config, job, organization)
        print("[OK] Job xác nhận Task ID và Organization; không cần quyền đọc Task cha.")
        labels = client.job_labels()
        if not labels:
            raise ValueError("Job chưa có label.")
        for label in labels:
            print(f"Job label: ID {label['id']} | {redactor.clean(label['name'])} | "
                  f"type={redactor.clean(label.get('type', 'any'))}")
        if args.audit_labels:
            print("[3/5] Loading model for label audit only...")
            detector = load_detector(settings)
            coverage = label_coverage(detector.names, labels, overrides, shape_types=detector_formats(detector))
            print_label_coverage(coverage, redactor)
            report_path = save_label_audit(config, detector.names, coverage, redactor)
            print(f"Báo cáo ID/type labels: {report_path}")
            print("[OK] Audit labels hoàn tất; không tải frame, không chạy suy luận ảnh, không ghi annotation.")
            return 0
        metadata = client.metadata()
        scope = JobScope(config.job_id, build_frames(job, metadata), {label["id"]: label for label in labels})
        existing = client.annotations()
        print(f"Valid Job frames: {len(scope.frames)}; labels: {len(labels)}")
        if args.convert_existing_to_boxes:
            from box_conversion import run_conversion
            return run_conversion(client, scope, job, settings, args)
        print("[3/5] Loading annotation source...")
        detector = load_detector(settings)
        formats = detector_formats(detector)
        mapper = resolve_label_mapping(detector.names, labels, overrides, redactor, shape_types=formats)
        scope.allowed_shape_types = set(formats.values())
        if settings.pipeline == "bdd-reference":
            scope.allowed_sources = {"file"}
        elif settings.pipeline == "scene-reference":
            scope.allowed_sources = {"auto", "file"}
        for class_id, label in mapper.mapping.items():
            geometry = formats.get(class_id, "rectangle")
            if label["id"] in scope.label_geometry and scope.label_geometry[label["id"]] != geometry:
                raise ValueError("Cùng label ID bị dùng cho hai geometry khác nhau; kiểm tra mapping.")
            scope.label_geometry[label["id"]] = geometry
        coverage = label_coverage(detector.names, labels, overrides, shape_types=formats)
        print_label_coverage(coverage, redactor)
        if args.require_all_labels:
            require_full_coverage(coverage)
        for class_id, label in mapper.mapping.items():
            print(f"Mapping: {redactor.clean(detector.names[class_id])} -> "
                  f"{redactor.clean(label['name'])} (ID {label['id']})")
            mapper.attributes(label)  # Stop early if label attributes cannot be represented safely.
        print("Lớp YOLO không ánh xạ:", redactor.clean(", ".join(mapper.unmapped)))
        mapped_ids = {label["id"] for label in mapper.mapping.values()}
        unsupported = [label["name"] for label in labels if label["id"] not in mapped_ids]
        print("Label CVAT chưa được model hỗ trợ:", redactor.clean(", ".join(unsupported)) or "none")
        print("Lane/drivable area/segmentation chỉ được xử lý bằng mô hình phù hợp riêng.")
        if settings.pipeline in ("bdd-reference", "scene-reference"):
            detector.prepare(client, job, metadata, scope)
        state = RunState({"url": config.url, "org": config.organization, "task": config.task_id,
                          "job": config.job_id, "user_id": user["id"], "weights": detector.weights_hash,
                          "settings": settings.checkpoint_dict(), "mapping": overrides, "labels": labels,
                          "metadata": metadata, "job_range": [job["start_frame"], job["stop_frame"]]})
        session = AnnotationSession(client=client, config=config, scope=scope, detector=detector,
                                    mapper=mapper, settings=settings, state=state, job=job, existing=existing)
        if args.all:
            return session.run_all(args.apply)
        result = session.run_stage(args.limit, args.apply)
        if result.completed:
            print_next_action(state, config, args)
        return result.code
    except DiagnosticError as error:
        print(f"[LỖI] {error.step} | HTTP {error.status or 'N/A'} | {error.detail}")
        print(explain_status(error.status, error.step))
        return 1
    except (ValueError, FileNotFoundError) as error:
        print("[DỪNG]", redactor.clean(error))
        return 1
    except (KeyboardInterrupt, EOFError):
        print("Đã hủy. Nếu đang ghi, hãy đọc lại annotation trước khi chạy tiếp.")
        return 2
    except Exception as error:
        print(f"[LỖI] {type(error).__name__}: dừng an toàn; exception gốc đã ẩn để tránh lộ bí mật.")
        return 1
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
