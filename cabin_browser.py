"""Execute a reviewed script in the user's open CVAT console, without extracting auth."""
import argparse
import json
import time
from pathlib import Path
from cvat_task6_desktop import focus
from desktop_cvat import u, type_text, hotkey, clipboard

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('script')
    p.add_argument('output')
    p.add_argument('--delay', type=float, default=4)
    args = p.parse_args()
    focus()
    u.SetCursorPos(1330, 980)
    u.mouse_event(2, 0, 0, 0, 0)
    u.mouse_event(4, 0, 0, 0, 0)
    hotkey(0x11, 0x23)
    time.sleep(.2)
    type_text(Path(args.script).read_text(encoding='utf-8'))
    hotkey(0x0D)
    time.sleep(args.delay)
    result = json.loads(clipboard())
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    print('Saved', args.output)
