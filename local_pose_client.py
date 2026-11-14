"""Use the interactive CVAT session without persisting any auth secrets."""
import json
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'outputs' / 'local_pose'


def request(path, *, method='GET', params=None, payload=None, binary=False, timeout=240):
    queue = ROOT / 'queue'
    queue.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex
    command = {'path': path, 'method': method, 'params': params,
               'json': payload, 'binary': binary}
    target = queue / (name + '.request.json')
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(command), encoding='utf-8')
    temp.replace(target)
    response = queue / (name + '.response.json')
    deadline = time.monotonic() + timeout
    while not response.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('CVAT request not confirmed. Inspect response before retrying writes: ' + name)
        time.sleep(.2)
    result = json.loads(response.read_text(encoding='utf-8'))
    if not 200 <= result.get('status', 0) < 300:
        raise RuntimeError(json.dumps(result))
    if result['status'] in (204, 205):
        return None
    return Path(result['file']).read_bytes() if binary else result['data']


def snapshot(job_id):
    folder = ROOT / f'job{job_id}'
    folder.mkdir(exist_ok=True)
    data = {}
    for name, endpoint, params in [
        ('job', f'/api/jobs/{job_id}', None),
        ('labels', '/api/labels', {'job_id': job_id, 'page_size': 100}),
        ('annotations', f'/api/jobs/{job_id}/annotations', None),
        ('meta', f'/api/jobs/{job_id}/data/meta', None),
    ]:
        data[name] = request(endpoint, params=params)
        (folder / (name + '.json')).write_text(json.dumps(data[name], ensure_ascii=False, indent=2), encoding='utf-8')
    return data
