"""Run a local reviewed JS file in the open CVAT DevTools console."""
import argparse
import ctypes as c
import json
import time
from pathlib import Path
from ctypes import wintypes as w
from cvat_task6_desktop import focus
from desktop_cvat import u, k, hotkey, clipboard


def paste(text):
    raw = (text + '\0').encode('utf-16-le')
    k.GlobalAlloc.argtypes = [w.UINT, c.c_size_t]
    k.GlobalAlloc.restype = w.HGLOBAL
    u.SetClipboardData.argtypes = [w.UINT, w.HANDLE]
    u.SetClipboardData.restype = w.HANDLE
    handle = k.GlobalAlloc(0x42, len(raw))
    ptr = k.GlobalLock(handle)
    c.memmove(ptr, raw, len(raw))
    k.GlobalUnlock(handle)
    if not u.OpenClipboard(None):
        raise RuntimeError('Clipboard busy')
    try:
        u.EmptyClipboard()
        if not u.SetClipboardData(13, handle):
            raise RuntimeError('Clipboard write failed')
    finally:
        u.CloseClipboard()
    hotkey(0x11, 0x56)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('script')
    p.add_argument('output')
    p.add_argument('--delay', type=float, default=3)
    a = p.parse_args()
    focus()
    u.SetCursorPos(1274, 337)
    u.mouse_event(2, 0, 0, 0, 0)
    u.mouse_event(4, 0, 0, 0, 0)
    time.sleep(.3)
    hotkey(0x11, 0x4C)
    time.sleep(.3)
    u.SetCursorPos(1274, 337)
    u.mouse_event(2, 0, 0, 0, 0)
    u.mouse_event(4, 0, 0, 0, 0)
    hotkey(0x11, 0x41)
    paste(' '.join(Path(a.script).read_text(encoding='utf-8').splitlines()))
    time.sleep(.3)
    hotkey(0x0D)
    time.sleep(a.delay)
    value = json.loads(clipboard())
    Path(a.output).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved', a.output)
