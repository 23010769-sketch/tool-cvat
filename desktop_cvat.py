"""Small local desktop controls for the user's open CVAT tab; no auth extraction."""
import argparse
import ctypes as c
from ctypes import wintypes as w
import json
from pathlib import Path
import time
from PIL import ImageGrab

u = c.windll.user32
k = c.windll.kernel32
u.SetProcessDPIAware()
u.GlobalLock = k.GlobalLock
k.GlobalLock.argtypes = [w.HGLOBAL]
k.GlobalLock.restype = c.c_void_p
k.GlobalUnlock.argtypes = [w.HGLOBAL]
u.GetClipboardData.argtypes = [w.UINT]
u.GetClipboardData.restype = w.HANDLE

class KEYBDINPUT(c.Structure):
    _fields_ = [('wVk', w.WORD), ('wScan', w.WORD), ('dwFlags', w.DWORD),
                ('time', w.DWORD), ('dwExtraInfo', c.c_size_t)]
class MOUSEINPUT(c.Structure):
    _fields_ = [('dx', w.LONG), ('dy', w.LONG), ('mouseData', w.DWORD),
                ('dwFlags', w.DWORD), ('time', w.DWORD), ('dwExtraInfo', c.c_size_t)]
class INPUTUNION(c.Union):
    _fields_ = [('ki', KEYBDINPUT), ('mi', MOUSEINPUT)]
class INPUT(c.Structure):
    _fields_ = [('type', w.DWORD), ('data', INPUTUNION)]

def key(code, up=False):
    flags = (2 if up else 0) | (1 if code in range(0x21, 0x2F) else 0)
    item = INPUT(type=1, data=INPUTUNION(ki=KEYBDINPUT(wVk=code, dwFlags=flags)))
    if u.SendInput(1, c.byref(item), c.sizeof(INPUT)) != 1:
        raise OSError('Keyboard input failed')

def hotkey(*codes):
    for code in codes: key(code)
    for code in reversed(codes): key(code, True)

def type_text(text):
    units = text.encode('utf-16-le')
    events = []
    for i in range(0, len(units), 2):
        code = int.from_bytes(units[i:i+2], 'little')
        events.extend([INPUT(type=1, data=INPUTUNION(ki=KEYBDINPUT(wScan=code, dwFlags=f))) for f in (4, 6)])
    for start in range(0, len(events), 100):
        part = events[start:start+100]
        array = (INPUT * len(part))(*part)
        if u.SendInput(len(part), array, c.sizeof(INPUT)) != len(part):
            raise OSError('Text input failed')
        time.sleep(.01)

def focus():
    found = []
    @c.WINFUNCTYPE(c.c_bool, w.HWND, w.LPARAM)
    def enum(h, p):
        b = c.create_unicode_buffer(1000)
        u.GetWindowTextW(h, b, 1000)
        if u.IsWindowVisible(h) and 'Computer Vision Annotation Tool - Google Chrome' in b.value:
            found.append(h)
        return True
    u.EnumWindows(enum, 0)
    select_tab = True
    if not found:
        known = 789082
        pid = w.DWORD()
        u.GetWindowThreadProcessId(known, c.byref(pid))
        if not u.IsWindow(known) or pid.value != 9064:
            raise RuntimeError('Known Chrome window is unavailable')
        found = [known]
        select_tab = True
    if len(found) != 1: raise RuntimeError('Expected one CVAT Chrome window')
    u.ShowWindow(found[0], 3)
    current = u.GetForegroundWindow()
    a = u.GetWindowThreadProcessId(current, None)
    b = k.GetCurrentThreadId()
    attached = bool(u.AttachThreadInput(b, a, True)) if a != b else False
    u.SetForegroundWindow(found[0])
    u.BringWindowToTop(found[0])
    if attached: u.AttachThreadInput(b, a, False)
    time.sleep(.6)
    if u.GetForegroundWindow() != found[0]:
        raise RuntimeError('CVAT is not foreground; no input sent')
    if select_tab:
        hotkey(0x11, 0x34)
        time.sleep(.6)
        title = c.create_unicode_buffer(1000)
        u.GetWindowTextW(found[0], title, 1000)
        if 'Computer Vision Annotation Tool' not in title.value:
            raise RuntimeError('CVAT tab not selected; no script sent')
    return found[0]

def clipboard():
    if not u.OpenClipboard(None): raise RuntimeError('Clipboard unavailable')
    try:
        handle=u.GetClipboardData(13)
        if not handle: raise RuntimeError('No clipboard text')
        ptr=k.GlobalLock(handle)
        try: return c.wstring_at(ptr)
        finally: k.GlobalUnlock(handle)
    finally: u.CloseClipboard()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--focus',action='store_true')
    p.add_argument('--console',action='store_true')
    p.add_argument('--script')
    p.add_argument('--hotkey',type=lambda s:[int(x,16) for x in s.split(',')])
    p.add_argument('--click',nargs=2,type=int)
    p.add_argument('--capture')
    p.add_argument('--clipboard-json')
    p.add_argument('--delay',type=float,default=.7)
    a=p.parse_args()
    expected = focus() if a.focus else None
    if a.console: hotkey(0x11,0x10,0x4A);time.sleep(1)
    if a.click:
        u.SetCursorPos(*a.click);u.mouse_event(2,0,0,0,0);u.mouse_event(4,0,0,0,0);time.sleep(.2)
    if a.hotkey: hotkey(*a.hotkey)
    if a.script:
        if not expected or u.GetForegroundWindow() != expected:
            raise RuntimeError('Use --focus; CVAT must remain foreground before typing')
        type_text(Path(a.script).read_text(encoding='utf-8'));hotkey(0x0D)
    time.sleep(a.delay)
    if a.capture: ImageGrab.grab().save(a.capture)
    if a.clipboard_json:
        data=json.loads(clipboard())
        Path(a.clipboard_json).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Saved JSON:',a.clipboard_json)

if __name__=='__main__': main()
