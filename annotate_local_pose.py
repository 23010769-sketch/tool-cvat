"""Draft and append COCO-17 skeletons using the user's interactive local session."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from frame_mapping import build_frames
from local_pose_client import ROOT, request, snapshot

NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
         'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
         'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
         'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
EDGES = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def normalize(name):
    return ''.join(c for c in name.lower() if c.isalnum())


def resolve(labels):
    candidates = [l for l in labels if l['type'] == 'skeleton' and len(l.get('sublabels', [])) == 17]
    if len(candidates) != 1:
        raise ValueError('Need exactly one skeleton label with 17 sublabels')
    parent = candidates[0]
    actual = {normalize(l['name']): l for l in parent['sublabels']}
    mapping = []
    for name in NAMES:
        aliases = {normalize(name)}
        if name.startswith(('left_', 'right_')):
            side, part = name.split('_', 1)
            aliases.update({side[0] + part, part + side, part + side[0]})
        found = [v for k, v in actual.items() if k in aliases]
        if len(found) != 1:
            raise ValueError(f'Cannot map {name}; live sublabels: {list(actual)}')
        mapping.append(found[0])
    if len({l['id'] for l in mapping}) != 17:
        raise ValueError('Keypoint mapping is not unique')
    return parent, mapping


def attributes(label):
    return [{'spec_id': a['id'], 'value': str(a['default_value'])} for a in label.get('attributes', [])]


def matches_existing(xy, scores, bbox, shape, mapping):
    elements = {e['label_id']: e for e in shape.get('elements', [])}
    distances, points = [], []
    for index, label in enumerate(mapping):
        element = elements.get(label['id'])
        if not element or element.get('outside') or len(element.get('points', [])) != 2:
            continue
        point = np.asarray(element['points'], dtype=float)
        points.append(point)
        if scores[index] >= .4:
            distances.append(float(np.linalg.norm(xy[index] - point)))
    if len(points) < 3:
        return False
    points = np.asarray(points)
    low, high = points.min(axis=0), points.max(axis=0)
    scale = max(float(np.linalg.norm(high-low)), 1)
    if len(distances) >= 3 and float(np.median(distances)) / scale < .15:
        return True
    bx1,by1,bx2,by2 = bbox
    intersection = max(0,min(high[0],bx2)-max(low[0],bx1))*max(0,min(high[1],by2)-max(low[1],by1))
    union = (bx2-bx1)*(by2-by1)+float(np.prod(high-low))-intersection
    return union > 0 and intersection/union > .5


def draft(job_id, limit=None, model_name='yolo11m-pose.pt', image_size=640):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(8)
    folder = ROOT / f'job{job_id}'
    data = snapshot(job_id)
    job, meta, existing = data['job'], data['meta'], data['annotations']
    if job['task_id'] != 4:
        raise ValueError('Current authorization is Task 4 only')
    frames = build_frames(job, meta)
    label_response = data['labels']
    if isinstance(label_response, dict) and label_response.get('next'):
        raise ValueError('Labels pagination needs handling')
    labels = label_response.get('results', []) if isinstance(label_response, dict) else label_response
    parent, mapping = resolve(labels)
    save(folder / 'keypoint_mapping.json', {'parent':parent['id'], 'mapping':[
        {'model_index': i, 'name': n, 'label_id': l['id']} for i,(n,l) in enumerate(zip(NAMES,mapping))]})
    if existing.get('tracks'):
        raise ValueError('Existing tracks require explicit duplicate analysis before adding skeletons')
    occupied = {s['frame'] for s in existing.get('shapes',[]) if s['label_id']==parent['id']}
    model = YOLO(str(Path(__file__).resolve().parent / 'models' / model_name))
    image_folder, review_folder = folder / 'images', folder / 'review'
    image_folder.mkdir(exist_ok=True); review_folder.mkdir(exist_ok=True)
    plan = {'job_id':job_id,'task_id':job['task_id'],'model':model_name,'image_size':image_size,
            'model_sha256':hashlib.sha256(Path(model.ckpt_path).read_bytes()).hexdigest(),
            'frames':[], 'shapes':[], 'existing_skeleton_frames':sorted(occupied)}
    selected = list(frames.items())[:limit] if limit else list(frames.items())
    for index, (number, info) in enumerate(selected):
        image_path = image_folder / f'{number:06d}.jpg'
        if not image_path.exists():
            image_path.write_bytes(request(f'/api/jobs/{job_id}/data',params={
                'type':'frame','number':number,'quality':'original'},binary=True))
        image = cv2.imread(str(image_path))
        if image is None or image.shape[:2] != (info.height, info.width):
            raise ValueError('Frame dimensions do not match metadata')
        current_shapes = [s for s in existing.get('shapes',[]) if s['frame']==number and s['label_id']==parent['id']]
        result = model.predict(image, imgsz=image_size, conf=.30, iou=.6, device='cpu', verbose=False)[0]
        people = []
        overlay = image.copy()
        for detection_index in range(len(result.boxes)):
            confidence = float(result.boxes.conf[detection_index])
            xy = result.keypoints.xy[detection_index].cpu().numpy()
            scores = result.keypoints.conf[detection_index].cpu().numpy()
            if xy.shape != (17,2) or not np.isfinite(xy).all() or not np.isfinite(scores).all():
                raise ValueError('Invalid model keypoints')
            if int((scores >= .5).sum()) < 4:
                continue
            box = result.boxes.xyxy[detection_index].cpu().tolist()
            if any(matches_existing(xy,scores,box,s,mapping) for s in current_shapes):
                people.append({'box':box,'score':confidence,'status':'preserved_existing'})
                continue
            elements=[]
            for (x,y), score, sublabel in zip(xy,scores,mapping):
                elements.append({'type':'points','frame':number,'label_id':sublabel['id'],
                    'points':[round(float(np.clip(x,0,info.width)),2),round(float(np.clip(y,0,info.height)),2)],
                    'outside':bool(score<.25),'occluded':False,'source':'auto',
                    'attributes':attributes(sublabel),'score':round(float(score),6)})
            shape={'type':'skeleton','frame':number,'label_id':parent['id'],'points':[],
                   'source':'auto','attributes':attributes(parent),'occluded':False,
                   'outside':False,'score':round(confidence,6),'elements':elements}
            plan['shapes'].append(shape)
            people.append({'box':box,
                'score':confidence,'keypoints':xy.tolist(),'confidence':scores.tolist()})
            for a,b in EDGES:
                if min(scores[a], scores[b])>=.25:
                    cv2.line(overlay,tuple(xy[a].astype(int)),tuple(xy[b].astype(int)),(50,240,50),2)
            for i,((x,y),score) in enumerate(zip(xy,scores)):
                if score>=.25:
                    point=(int(x),int(y))
                    color=(50,240,50) if score>=.5 else (0,180,255)
                    cv2.circle(overlay,point,3,color,-1)
                    cv2.putText(overlay,str(i),point,cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),2)
                    cv2.putText(overlay,str(i),point,cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
        cv2.putText(overlay,f'Frame {number} | {len(people)} people',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
        cv2.imwrite(str(review_folder / f'{number:06d}.jpg'),overlay)
        plan['frames'].append({'number':number,'status':'predicted','people':people,
            'image_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest()})
        save(folder / ('sample_plan.json' if limit else 'plan.json'),plan)
        print(f'Frame {number}: {len(people)} skeletons ({index+1}/{len(selected)})',flush=True)
    save(folder / ('sample_plan.json' if limit else 'plan.json'),plan)
    print(f'Draft ready: {len(plan["shapes"])} skeletons, {len(selected)} frames',flush=True)


def canonical(shape):
    result=copy.deepcopy(shape)
    result.pop('id',None)
    for element in result.get('elements',[]):
        element.pop('id',None)
    return result


def verify_preserved(before, after):
    def normalized(value):
        if isinstance(value,dict):
            return {key:sorted((normalized(e) for e in items),key=lambda e:e.get('id',e.get('spec_id',0)))
                    if key in ('elements','attributes') else normalized(items) for key,items in value.items()}
        if isinstance(value,list):
            return [normalized(item) for item in value]
        return value
    for key in ['shapes','tracks','tags','intervals']:
        current={item['id']:item for item in after.get(key,[])}
        for item in before.get(key,[]):
            if normalized(current.get(item['id'])) != normalized(item):
                raise RuntimeError('Pre-existing annotation changed; stop writing')


def apply(job_id):
    folder=ROOT/f'job{job_id}'
    plan=json.loads((folder/'plan.json').read_text(encoding='utf-8'))
    baseline=json.loads((folder/'annotations.json').read_text(encoding='utf-8'))
    labels_snapshot=json.loads((folder/'labels.json').read_text(encoding='utf-8'))
    job=request(f'/api/jobs/{job_id}')
    if job['task_id'] != 4 or plan['task_id'] != 4 or plan['job_id'] != job_id:
        raise RuntimeError('Write scope must be Task 4 / selected Job')
    metadata=request(f'/api/jobs/{job_id}/data/meta')
    if metadata != json.loads((folder/'meta.json').read_text(encoding='utf-8')):
        raise RuntimeError('Image metadata changed')
    frame_scope=build_frames(job,metadata)
    label_list=labels_snapshot.get('results',[]) if isinstance(labels_snapshot,dict) else labels_snapshot
    parent,mapping=resolve(label_list)
    for shape in plan['shapes']:
        if shape.get('id') is not None or shape['label_id'] != parent['id'] or shape['type'] != 'skeleton' or shape['frame'] not in frame_scope:
            raise ValueError('Invalid skeleton scope')
        if len(shape['elements']) != 17 or {e['label_id'] for e in shape['elements']} != {l['id'] for l in mapping}:
            raise ValueError('Skeleton must contain exactly 17 distinct mapped keypoints')
        info=frame_scope[shape['frame']]
        for element in shape['elements']:
            if element.get('id') is not None or element['type']!='points' or element['frame']!=shape['frame'] or len(element['points'])!=2:
                raise ValueError('Invalid keypoint')
            x,y=element['points']
            if not math.isfinite(x) or not math.isfinite(y) or not (0<=x<=info.width and 0<=y<=info.height):
                raise ValueError('Keypoint outside image dimensions')
    live=request(f'/api/jobs/{job_id}/annotations')
    if request('/api/labels',params={'job_id':job_id,'page_size':100}) != labels_snapshot:
        raise RuntimeError('Label schema changed')
    verify_preserved(baseline,live)
    save(folder/'before_apply.json',live)
    # Resume only exact predictions; never resend an uncertain write blindly.
    def signature(shape):
        return (shape['frame'],shape['label_id'],tuple(sorted((e['label_id'],tuple(round(float(p),2) for p in e['points']),e.get('outside',False)) for e in shape.get('elements',[]))))
    known={signature(s) for s in live['shapes'] if s['type']=='skeleton'}
    baseline_ids={s['id'] for s in baseline['shapes']}
    planned={signature(s) for s in plan['shapes']}
    if any(s['id'] not in baseline_ids and (s['type']!='skeleton' or signature(s) not in planned) for s in live['shapes']):
        raise RuntimeError('Unreviewed annotations appeared since drafting; stop to prevent duplicates')
    pending=[s for s in plan['shapes'] if signature(s) not in known]
    first=True
    while pending:
        batch=pending[:1 if first else 50]
        pending=pending[len(batch):]
        first=False
        save(folder/'pending_write.json',{'version':live['version'],'shapes':batch})
        response=request(f'/api/jobs/{job_id}/annotations',method='PATCH',params={'action':'create'},
            payload={'version':live['version'],'shapes':batch,'tags':[],'tracks':[]})
        after=request(f'/api/jobs/{job_id}/annotations')
        verify_preserved(live,after)
        previous_ids={s['id'] for s in live['shapes']}
        added=[s for s in after['shapes'] if s['id'] not in previous_ids]
        if sorted(map(signature,added)) != sorted(map(signature,batch)):
            raise RuntimeError('Created skeletons differ from requested predictions; stop writing')
        expected_by_signature={signature(s):s for s in batch}
        for created in added:
            expected=expected_by_signature[signature(created)]
            if len(created['elements'])!=17:
                raise RuntimeError('Server did not preserve 17 keypoints')
            actual_elements={e['label_id']:e for e in created['elements']}
            for point in expected['elements']:
                actual=actual_elements[point['label_id']]
                for key in ['frame','type','outside','occluded','source','attributes']:
                    if actual[key] != point[key]:
                        raise RuntimeError('Server returned different keypoint fields')
        live=after
        save(folder/'verified_annotations.json',after)
        save(folder/'apply_status.json',{'verified_skeletons':len(live['shapes']),'remaining':len(pending)})
        print(f'Verified {len(added)} new skeletons; {len(pending)} remaining',flush=True)
    # The final audit also covers a batch that succeeded before a local
    # interruption, without sending any create request again.
    verify_preserved(baseline,live)
    by_signature={signature(s):s for s in live['shapes'] if s['type']=='skeleton'}
    for expected in plan['shapes']:
        actual=by_signature.get(signature(expected))
        if actual is None or len(actual.get('elements',[]))!=17:
            raise RuntimeError('Final audit is missing planned skeleton/keypoints')
        actual_elements={e['label_id']:e for e in actual['elements']}
        for point in expected['elements']:
            returned=actual_elements[point['label_id']]
            for key in ['frame','type','outside','occluded','source','attributes']:
                if returned[key]!=point[key]:
                    raise RuntimeError('Final audit found different keypoint fields')
    save(folder/'verified_annotations.json',live)
    save(folder/'apply_status.json',{'state':'verified','job_id':job_id,'task_id':4,
        'planned_skeletons_verified':len(plan['shapes']),'total_skeletons':len(live['shapes']),
        'total_keypoints':sum(len(s.get('elements',[])) for s in live['shapes']),
        'frames':len(plan['frames']),'remaining':0,'existing_preserved':True})
    print('All planned skeletons verified on CVAT.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=int,default=4)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--model',default='yolo11m-pose.pt')
    parser.add_argument('--imgsz',type=int,default=640)
    args=parser.parse_args()
    if args.apply:
        apply(args.job)
    else:
        draft(args.job,args.limit,args.model,args.imgsz)
