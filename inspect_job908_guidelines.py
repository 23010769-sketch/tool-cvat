"""Read-only inspection of Job 908; credentials are entered locally, never saved."""

import io
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk

from PIL import Image

from annotation_inventory import summarize_annotations
from auth import Credentials, DiagnosticError, Redactor, read_credentials
from config import Config
from cvat_client import CvatClient
from frame_mapping import build_frames, validate_job


OUTPUT = Path(__file__).resolve().parent / "outputs" / "job908_review"


def inspect(username, password, report):
    """Only login and GET requests; no annotation or workflow mutations."""
    redactor = Redactor()
    redactor.add(username, password)
    config = Config(task_id=211, job_id=908, backend="rest", frame_timeout=180)
    client = CvatClient(config, redactor)
    try:
        client.login(Credentials("password", username=username, password=password))
        password = ""
        user = client.current_user()
        client.set_organization()
        job = client.job()
        organization = client.organization(job["organization"])
        validate_job(config, job, organization)
        assignee = job.get("assignee") or {}
        if assignee.get("id") != user["id"]:
            raise ValueError("Job 908 không được giao cho tài khoản vừa đăng nhập; dừng.")
        labels = client.job_labels()
        metadata = client.metadata()
        frames = build_frames(job, metadata)
        annotations = client.annotations()
        issues = client._list("Issues của Job 908", None, "/api/issues", {"job_id": 908})
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, data in (("job", job), ("labels", labels), ("metadata", metadata),
                           ("annotations_before", annotations), ("issues", issues),
                           ("inventory", summarize_annotations(labels, annotations))):
            path = OUTPUT / (name + ".json")
            if name == "annotations_before" and path.exists():
                # Preserve the first snapshot across repeated inspections.
                path = OUTPUT / "annotations_latest.json"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        for index, (number, info) in enumerate(sorted(frames.items()), 1):
            report(f"Đang đọc ảnh {index}/{len(frames)} — frame {number}")
            raw = client.frame_bytes(number, frames)
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                if image.size != (info.width, info.height):
                    raise ValueError("Kích thước ảnh khác metadata; dừng.")
                image.convert("RGB").save(OUTPUT / f"frame_{number:06d}.png")
        result = {"job_id": 908, "task_id": 211, "frames": sorted(frames),
                  "shapes": len(annotations.get("shapes", [])), "issues": len(issues),
                  "read_only": True}
        (OUTPUT / "complete.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        report(f"Đã đọc {len(frames)} ảnh và nhãn hiện tại. Chưa sửa dữ liệu CVAT.")
        return True
    except DiagnosticError as error:
        report(f"Dừng: {error.step} — HTTP {error.status or 'N/A'} — {redactor.clean(error.detail)}")
    except ValueError as error:
        report("Dừng: " + redactor.clean(error))
    except Exception as error:
        report(f"Dừng do lỗi {type(error).__name__}; không in thông tin đăng nhập.")
    finally:
        client.close()
    return False


def gui_main():
    root = tk.Tk()
    root.title("CVAT — kiểm tra Job 908 theo guideline")
    root.geometry("600x310")
    panel = ttk.Frame(root, padding=18)
    panel.pack(fill="both", expand=True)
    ttk.Label(panel, text="Đăng nhập CVAT trên máy để đọc Job 908", font=("Segoe UI", 12)).pack(anchor="w")
    ttk.Label(panel, text="cohort-5a / Task 211 / Job 908\nChỉ đọc ảnh và nhãn. Mật khẩu không lưu vào file.").pack(anchor="w", pady=8)
    username = tk.StringVar()
    password = tk.StringVar()
    ttk.Label(panel, text="Username / MSSV").pack(anchor="w")
    user_entry = ttk.Entry(panel, textvariable=username, width=50)
    user_entry.pack(anchor="w")
    ttk.Label(panel, text="Mật khẩu").pack(anchor="w", pady=(6, 0))
    password_entry = ttk.Entry(panel, textvariable=password, show="*", width=50)
    password_entry.pack(anchor="w")
    status = tk.StringVar(value="Nhập tài khoản tại đây, không gửi mật khẩu vào chat.")
    events = queue.Queue()

    def submit():
        if not username.get().strip() or not password.get():
            status.set("Vui lòng nhập đủ username và mật khẩu.")
            return
        values = username.get().strip(), password.get()
        password.set("")
        button.configure(state="disabled")
        user_entry.configure(state="disabled")
        password_entry.configure(state="disabled")
        status.set("Đang đăng nhập và kiểm tra đúng Job...")

        def run():
            ok = inspect(*values, events.put)
            events.put(("done", ok))

        threading.Thread(target=run, daemon=True).start()

    button = ttk.Button(panel, text="Đăng nhập và kiểm tra", command=submit)
    button.pack(anchor="w", pady=10)
    ttk.Label(panel, textvariable=status, wraplength=560).pack(anchor="w")

    def poll():
        try:
            while True:
                item = events.get_nowait()
                if isinstance(item, tuple):
                    if not item[1]:
                        button.configure(state="normal")
                        user_entry.configure(state="normal")
                        password_entry.configure(state="normal")
                else:
                    status.set(item)
        except queue.Empty:
            pass
        root.after(200, poll)

    user_entry.focus_set()
    root.bind("<Return>", lambda event: submit() if str(button["state"]) != "disabled" else None)
    poll()
    root.mainloop()


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Đăng nhập tại Terminal và chỉ đọc Job 908 để kiểm tra guideline.")
    parser.add_argument("--gui", action="store_true", help="Dùng cửa sổ đăng nhập thay cho Terminal.")
    args = parser.parse_args(argv)
    if args.gui:
        gui_main()
        return 0
    print("CVAT — Task 211 / Job 908 / cohort-5a — CHỈ ĐỌC")
    print("Nhập tài khoản tại Terminal; mật khẩu được ẩn và không lưu vào file.")
    redactor = Redactor()
    try:
        credentials = read_credentials("password", None, redactor)
        return 0 if inspect(credentials.username, credentials.password, print) else 1
    except (ValueError, EOFError, KeyboardInterrupt) as error:
        print("[DỪNG]", redactor.clean(error) or "Đã hủy đăng nhập.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
