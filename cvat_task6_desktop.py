"""Operate the already open local CVAT window without reading credentials."""
import argparse
import ctypes as c
import json
import time
from pathlib import Path

from PIL import ImageGrab
from desktop_cvat import hotkey, type_text, clipboard, u, k


def focus():
    found = []
    from ctypes import wintypes as w
    @c.WINFUNCTYPE(c.c_bool, w.HWND, w.LPARAM)
    def enum(h, p):
        title = c.create_unicode_buffer(1000)
        u.GetWindowTextW(h, title, 1000)
        if u.IsWindowVisible(h) and 'Computer Vision Annotation Tool - Google Chrome' in title.value:
            found.append(h)
        return True
    u.EnumWindows(enum, 0)
    if len(found) != 1:
        raise RuntimeError('Expected one CVAT browser window')
    h = found[0]
    u.ShowWindow(h, 3)
    current = u.GetForegroundWindow()
    a, b = u.GetWindowThreadProcessId(current, None), k.GetCurrentThreadId()
    attached = bool(u.AttachThreadInput(b, a, True)) if a != b else False
    u.SetForegroundWindow(h)
    u.BringWindowToTop(h)
    if attached:
        u.AttachThreadInput(b, a, False)
    time.sleep(.6)
    if u.GetForegroundWindow() != h:
        raise RuntimeError('CVAT must be foreground')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--console', action='store_true')
    p.add_argument('--script')
    p.add_argument('--capture')
    p.add_argument('--clipboard-json')
    p.add_argument('--delay', type=float, default=1)
    a = p.parse_args()
    focus()
    if a.console:
        hotkey(0x11, 0x10, 0x4A)
        time.sleep(1)
    if a.script:
        type_text(Path(a.script).read_text(encoding='utf-8'))
        hotkey(0x0D)
    time.sleep(a.delay)
    if a.capture:
        ImageGrab.grab().save(a.capture)
    if a.clipboard_json:
        data = json.loads(clipboard())
        Path(a.clipboard_json).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Saved', a.clipboard_json)
