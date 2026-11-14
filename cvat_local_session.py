"""Interactive localhost CVAT session; credentials stay only in process memory."""
import getpass
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

ROOT = Path(__file__).resolve().parent / 'outputs' / 'local_pose'
BASE = 'http://localhost:8081'


def gui_credentials():
    import tkinter as tk
    from tkinter import ttk
    window = tk.Tk()
    window.title('Công cụ gán nhãn CVAT — Đăng nhập API')
    window.geometry('510x300+360+200')
    window.resizable(False, False)
    window.attributes('-topmost', True)
    panel = ttk.Frame(window, padding=24)
    panel.pack(fill='both', expand=True)
    ttk.Label(panel, text='Kết nối công cụ gán keypoint', font=('Segoe UI', 13, 'bold')).pack(anchor='w')
    ttk.Label(panel, text='CVAT local: http://localhost:8081\nDùng tài khoản CVAT của bạn. Mật khẩu không được lưu.',
              padding=(0, 8, 0, 10)).pack(anchor='w')
    ttk.Label(panel, text='Tên đăng nhập').pack(anchor='w')
    username = ttk.Entry(panel, width=48)
    username.pack(anchor='w', pady=(2, 8))
    username.insert(0, 'admin')
    ttk.Label(panel, text='Mật khẩu').pack(anchor='w')
    password = ttk.Entry(panel, show='●', width=48)
    password.pack(anchor='w', pady=(2, 12))
    values = []
    write_json(ROOT / 'session_status.json', {'state': 'waiting_for_user_login', 'base': BASE})
    def submit(event=None):
        if username.get().strip() and password.get():
            values.extend([username.get().strip(), password.get()])
            password.delete(0, 'end')
            window.destroy()
    ttk.Button(panel, text='Kết nối và tiếp tục gán nhãn', command=submit).pack(anchor='e')
    window.bind('<Return>', submit)
    password.focus_set()
    window.lift()
    window.mainloop()
    if not values:
        write_json(ROOT / 'session_status.json', {'state': 'login_dialog_closed', 'base': BASE})
        raise KeyboardInterrupt
    return values


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    queue = ROOT / 'queue'
    queue.mkdir(exist_ok=True)
    print('CVAT LOCAL - http://localhost:8081')
    print('Enter your CVAT account here. Password is hidden and is never saved.')
    if '--gui' in sys.argv:
        username, password = gui_credentials()
    else:
        username = input('Username: ').strip()
        password = getpass.getpass('Password: ')
    session = requests.Session()
    session.trust_env = False
    session.headers.update({'Referer': BASE + '/', 'Origin': BASE})
    try:
        response = session.post(BASE + '/api/auth/login', json={
            'username': username, 'password': password}, timeout=20, allow_redirects=False)
        password = None
        if response.status_code != 200:
            write_json(ROOT / 'session_status.json', {'state': 'login_failed', 'http': response.status_code})
            print('Login failed. HTTP', response.status_code)
            if '--gui' in sys.argv:
                from tkinter import messagebox
                messagebox.showerror('CVAT', f'Đăng nhập chưa thành công (HTTP {response.status_code}).')
            return
        token = response.json().get('key')
        if token:
            session.headers['Authorization'] = 'Token ' + token
        if session.cookies.get('csrftoken'):
            session.headers['X-CSRFToken'] = session.cookies.get('csrftoken')
        del response, token
        write_json(ROOT / 'session_status.json', {'state': 'ready', 'base': BASE})
        print('Connected. Keep this window open while annotation runs. Ctrl+C to stop.')
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            for command_path in sorted(queue.glob('*.request.json')):
                result_path = command_path.with_name(command_path.name.replace('.request.json', '.response.json'))
                if result_path.exists():
                    continue
                try:
                    command = json.loads(command_path.read_text(encoding='utf-8'))
                    path = command['path']
                    parsed = urlsplit(path)
                    if parsed.scheme or parsed.netloc or not re.fullmatch(
                        r'/api/(?:jobs(?:/\d+(?:/(?:annotations|data(?:/meta)?|issues))?)?|tasks(?:/\d+(?:/(?:annotations|data/meta))?)?|labels|server/schema)/?', parsed.path
                    ):
                        raise ValueError('Endpoint outside annotation scope')
                    method = command.get('method', 'GET').upper()
                    if method not in {'GET', 'PATCH'}:
                        raise ValueError('Unsupported method')
                    if method == 'PATCH' and not re.fullmatch(r'/api/jobs/\d+/annotations/?', parsed.path):
                        raise ValueError('Only Job annotation PATCH is supported')
                    response = session.request(method, BASE + path,
                        params=command.get('params'), json=command.get('json'),
                        timeout=(10, 180), allow_redirects=False)
                    result = {'status': response.status_code}
                    if command.get('binary') and response.ok:
                        target = ROOT / (command_path.stem.replace('.request', '') + '.bin')
                        target.write_bytes(response.content)
                        result['file'] = str(target)
                    else:
                        try:
                            result['data'] = response.json()
                        except ValueError:
                            result['error'] = 'Non-JSON response'
                    write_json(result_path, result)
                    print(method, parsed.path, response.status_code, flush=True)
                except Exception as error:
                    write_json(result_path, {'error': type(error).__name__})
            time.sleep(0.3)
    finally:
        session.close()
        write_json(ROOT / 'session_status.json', {'state': 'closed'})


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print('Session error:', type(error).__name__)
    if '--gui' not in sys.argv:
        input('Press Enter to close...')
