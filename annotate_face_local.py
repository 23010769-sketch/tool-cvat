"""Task 5 facial annotation. Run in the network-restricted command sandbox.

Private images, coordinates, errors and previews are saved locally only. Console
output contains aggregate counts, never image content, coordinates or API data.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / 'outputs' / 'face_runtime'))
os.environ.setdefault('GLOG_minloglevel', '2')

from frame_mapping import build_frames
from local_pose_client import ROOT, request

FOLDER = ROOT / 'job5_private'
MODEL = Path(__file__).resolve().parent / 'models' / 'face_landmarker.task'
MODEL_SHA = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
# Anatomical left/right, outer vermilion border for lip_upper/lip_lower.
LANDMARKS = {
    'left_eye_outer': 263, 'left_eye_inner': 362,
    'left_eye_upper': 386, 'left_eye_lower': 374,
    'right_eye_outer': 33, 'right_eye_inner': 133,
    'right_eye_upper': 159, 'right_eye_lower': 145,
    'nose_tip': 1, 'mouth_left': 291, 'mouth_right': 61,
    'lip_upper': 0, 'lip_lower': 17,
}
EDGES = [('left_eye_outer','left_eye_upper'),('left_eye_upper','left_eye_inner'),
         ('left_eye_inner','left_eye_lower'),('left_eye_lower','left_eye_outer'),
         ('right_eye_outer','right_eye_upper'),('right_eye_upper','right_eye_inner'),
         ('right_eye_inner','right_eye_lower'),('right_eye_lower','right_eye_outer'),
         ('mouth_left','lip_upper'),('lip_upper','mouth_right'),
         ('mouth_right','lip_lower'),('lip_lower','mouth_left')]


def save(name, value):
    FOLDER.mkdir(parents=True, exist_ok=True)
    path = FOLDER / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def normalized(value):
    if isinstance(value, dict):
        result = {k: normalized(v) for k,v in value.items()}
        for key in ['elements','attributes']:
            if key in result:
                result[key].sort(key=lambda v:v.get('id', v.get('spec_id', 0)))
        return result
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def preserve(before, after):
    for key in ['shapes','tracks','tags','intervals']:
        actual = {item['id']:item for item in after.get(key, [])}
        for item in before.get(key, []):
            if normalized(actual.get(item['id'])) != normalized(item):
                raise ValueError('An existing annotation changed')


def resolve_labels(response):
    if response.get('next'):
        raise ValueError('Unexpected label pagination')
    parents = [l for l in response['results'] if l['name']=='face' and l['type']=='skeleton']
    if len(parents)!=1:
        raise ValueError('Expected one face skeleton')
    parent = parents[0]
    children = {l['name']:l for l in parent['sublabels']}
    if set(children)!=set(LANDMARKS) or len(parent['sublabels'])!=13:
        raise ValueError('Face schema does not match the 13 named landmarks')
    if any(l['type']!='points' for l in children.values()):
        raise ValueError('Children must be points')
    return parent, children


def default_attributes(label):
    return [{'spec_id':a['id'],'value':str(a['default_value'])} for a in label.get('attributes', [])]


def create_detector():
    import mediapipe as mp
    if hashlib.sha256(MODEL.read_bytes()).hexdigest()!=MODEL_SHA:
        raise ValueError('Model checksum mismatch')
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=4, min_face_detection_confidence=.5,
        min_face_presence_confidence=.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False)
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def signature(shape):
    return (shape['frame'], shape['label_id'], shape['type'], tuple(sorted(
        (e['label_id'],tuple(round(float(v),2) for v in e['points']),e.get('outside',False),e.get('occluded',False))
        for e in shape.get('elements', []))))


def same_face(points, shape, children):
    import numpy as np
    reverse = {l['id']:n for n,l in children.items()}
    distances=[]
    for element in shape['elements']:
        if element['label_id'] in reverse and not element.get('outside', False):
            distances.append(float(np.linalg.norm(np.array(points[reverse[element['label_id']]])-element['points'])))
    array=np.array(list(points.values()))
    size=max(float(np.linalg.norm(array.max(axis=0)-array.min(axis=0))),1)
    return len(distances)>=3 and float(np.median(distances))/size<.25


def build_shape(frame, points, parent, children):
    return {'type':'skeleton','frame':frame,'label_id':parent['id'],'points':[],
        'source':'auto','outside':False,'occluded':False,'attributes':default_attributes(parent),
        'elements':[{'type':'points','frame':frame,'label_id':children[name]['id'],
            'points':points[name],'source':'auto','outside':False,'occluded':False,
            'attributes':default_attributes(children[name])} for name in LANDMARKS]}


def make_preview(image, shapes, children, target):
    import cv2
    by_id = {l['id']:n for n,l in children.items()}
    for shape in shapes:
        pts={by_id[e['label_id']]:tuple(round(v) for v in e['points']) for e in shape['elements']
             if e['label_id'] in by_id and not e.get('outside',False)}
        for a,b in EDGES:
            if a in pts and b in pts:
                cv2.line(image,pts[a],pts[b],(50,220,50),1)
        for point in pts.values():
            cv2.circle(image,point,3,(0,0,255),-1)
    cv2.imwrite(str(target),image)


def draft():
    import cv2
    import mediapipe as mp
    import numpy as np
    job=request('/api/jobs/5')
    if job['id']!=5 or job['task_id']!=5 or job['type']!='annotation' or job['mode']!='annotation' or job.get('replicas_count',0):
        raise ValueError('Unexpected job scope')
    labels=request('/api/labels',params={'job_id':5,'page_size':100})
    parent, children=resolve_labels(labels)
    meta=request('/api/jobs/5/data/meta')
    frames=build_frames(job,meta)
    before=request('/api/jobs/5/annotations')
    if before.get('tracks'):
        raise ValueError('Existing tracks need separate duplicate analysis')
    # One untouched baseline per run, plus the original record on first use.
    if not (FOLDER/'original_annotations.json').exists():
        save('original_annotations.json',before)
    save('baseline_annotations.json',before)
    save('labels.json',labels)
    save('meta.json',meta)
    save('job.json',job)
    plan={'job_id':5,'task_id':5,'model_sha256':MODEL_SHA,'mapping':LANDMARKS,
          'frames':[],'shapes':[],'review_required':True}
    review=FOLDER/'review'; review.mkdir(parents=True,exist_ok=True)
    detector=create_detector()
    try:
        for number,info in frames.items():
            raw=request('/api/jobs/5/data',params={'type':'frame','number':number,'quality':'original'},binary=True)
            image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None or image.shape[:2]!=(info.height,info.width):
                raise ValueError('Image dimensions differ from metadata')
            result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))))
            existing=[s for s in before['shapes'] if s['frame']==number and s['label_id']==parent['id']]
            new=[]
            uncertain=[]
            for landmarks in result.face_landmarks:
                points={name:[round(float(landmarks[index].x)*info.width,2),
                              round(float(landmarks[index].y)*info.height,2)] for name,index in LANDMARKS.items()}
                if not all(math.isfinite(x) and math.isfinite(y) and 0<=x<=info.width and 0<=y<=info.height for x,y in points.values()):
                    uncertain.append('Landmarks outside image bounds; prediction not written')
                    continue
                if any(same_face(points,shape,children) for shape in existing):
                    continue
                # General geometric checks catch inverted eye/mouth mappings.
                inter_eye=math.dist(points['left_eye_outer'],points['right_eye_outer'])
                if inter_eye<8 or math.dist(points['mouth_left'],points['mouth_right'])<3:
                    uncertain.append('Face too small or unreliable for this 13-point template')
                    continue
                new.append(build_shape(number,points,parent,children))
            plan['shapes'].extend(new)
            plan['frames'].append({'frame':number,'existing':len(existing),'new':len(new),
                'detected':len(result.face_landmarks),'warnings':uncertain,
                'image_sha256':hashlib.sha256(raw).hexdigest()})
            make_preview(image.copy(),existing+new,children,review/f'{number:06d}.jpg')
            save('plan.json',plan)
        save('draft_summary.json',{'frames':len(frames),'new_faces':len(plan['shapes']),
            'existing_faces':sum(s['label_id']==parent['id'] for s in before['shapes']),
            'uncovered_frames':sum(not f['existing'] and not f['new'] for f in plan['frames']),
            'warning_frames':sum(bool(f['warnings']) for f in plan['frames'])})
        print('DRAFT_READY',json.dumps(json.loads((FOLDER/'draft_summary.json').read_text())),flush=True)
    finally:
        detector.close()


def verify_predictions(expected_shapes,actual_shapes):
    available={signature(s):s for s in actual_shapes}
    for expected in expected_shapes:
        actual=available.get(signature(expected))
        if actual is None or len(actual.get('elements',[]))!=13:
            raise ValueError('Readback does not contain the complete requested face')
        children={e['label_id']:e for e in actual['elements']}
        for element in expected['elements']:
            actual_element=children[element['label_id']]
            for key in ['frame','type','outside','occluded','source','attributes']:
                if actual_element[key]!=element[key]:
                    raise ValueError('Readback keypoint fields differ')


def apply():
    plan=json.loads((FOLDER/'plan.json').read_text(encoding='utf-8'))
    baseline=json.loads((FOLDER/'baseline_annotations.json').read_text(encoding='utf-8'))
    labels=json.loads((FOLDER/'labels.json').read_text(encoding='utf-8'))
    meta=json.loads((FOLDER/'meta.json').read_text(encoding='utf-8'))
    job=request('/api/jobs/5')
    if job['task_id']!=5 or plan['task_id']!=5 or plan['job_id']!=5:
        raise ValueError('Wrong Task/Job scope')
    if request('/api/labels',params={'job_id':5,'page_size':100})!=labels or request('/api/jobs/5/data/meta')!=meta:
        raise ValueError('Job schema/media changed')
    parent,children=resolve_labels(labels)
    frames=build_frames(job,meta)
    for shape in plan['shapes']:
        if shape.get('id') is not None or shape['type']!='skeleton' or shape['label_id']!=parent['id'] or shape['frame'] not in frames:
            raise ValueError('Invalid proposed face')
        if len(shape['elements'])!=13 or {e['label_id'] for e in shape['elements']}!={l['id'] for l in children.values()}:
            raise ValueError('Must have exactly 13 mapped keypoints')
        info=frames[shape['frame']]
        for element in shape['elements']:
            x,y=element['points']
            if element.get('id') is not None or element['frame']!=shape['frame'] or element['type']!='points' or not (math.isfinite(x) and math.isfinite(y) and 0<=x<=info.width and 0<=y<=info.height):
                raise ValueError('Invalid point geometry')
    live=request('/api/jobs/5/annotations')
    preserve(baseline,live)
    baseline_ids={s['id'] for s in baseline['shapes']}
    planned={signature(s) for s in plan['shapes']}
    if any(s['id'] not in baseline_ids and signature(s) not in planned for s in live['shapes']):
        raise ValueError('New unreviewed annotations appeared; stop to avoid duplicates')
    known={signature(s) for s in live['shapes']}
    pending=[s for s in plan['shapes'] if signature(s) not in known]
    save('before_apply.json',live)
    while pending:
        batch=pending[:1]
        pending=pending[1:]
        save('pending_write.json',{'version':live['version'],'shapes':batch})
        request('/api/jobs/5/annotations',method='PATCH',params={'action':'create'},
            payload={'version':live['version'],'shapes':batch,'tracks':[],'tags':[]})
        after=request('/api/jobs/5/annotations')
        preserve(live,after)
        old_ids={s['id'] for s in live['shapes']}
        added=[s for s in after['shapes'] if s['id'] not in old_ids]
        if len(added)!=len(batch):
            raise ValueError('Unexpected number of annotations added')
        verify_predictions(batch,added)
        live=after
        save('verified_annotations.json',live)
    preserve(baseline,live)
    verify_predictions(plan['shapes'],live['shapes'])
    summary={'state':'verified','frames':len(frames),'new_faces_verified':len(plan['shapes']),
        'total_faces':sum(s['label_id']==parent['id'] for s in live['shapes']),
        'keypoints_per_face':13,'existing_preserved':True,'review_required':True,
        'uncovered_frames':sum(not any(s['frame']==n and s['label_id']==parent['id'] for s in live['shapes']) for n in frames)}
    save('apply_status.json',summary)
    print('CVAT_VERIFIED',json.dumps(summary),flush=True)


def smoke_test():
    import mediapipe as mp
    import numpy as np
    detector=create_detector()
    try:
        result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.zeros((256,256,3),dtype=np.uint8)))
        if result.face_landmarks:
            raise ValueError('Unexpected face in a synthetic blank image')
        print('LOCAL_MODEL_SMOKE_TEST_OK',flush=True)
    finally:
        detector.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    try:
        if args.smoke_test:
            smoke_test()
        elif args.apply:
            apply()
        else:
            draft()
    except Exception as error:
        save('last_error_private.json',{'type':type(error).__name__,'detail':str(error)})
        print('LOCAL_OPERATION_STOPPED',type(error).__name__,'Details saved locally; no private data printed.',flush=True)
        raise SystemExit(1)
