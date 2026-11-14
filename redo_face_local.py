"""Replace only Job 5 face skeletons after verifying a complete local draft.

Never print private images, coordinates or raw API/error responses.
"""
import argparse
import copy
import datetime
import json
import math
from pathlib import Path

from annotate_face_local import (
    FOLDER, LANDMARKS, MODEL_SHA, build_shape, create_detector,
    default_attributes, make_preview, normalized, preserve,
    request, resolve_labels, signature, verify_predictions,
)
from frame_mapping import build_frames

POINTER = FOLDER / 'redo_current.json'
CONTOURS = {
    'left_eye_upper': [263,466,388,387,386,385,384,398,362],
    'left_eye_lower': [263,249,390,373,374,380,381,382,362],
    'right_eye_upper': [33,246,161,160,159,158,157,173,133],
    'right_eye_lower': [33,7,163,144,145,153,154,155,133],
    'lip_upper': [61,185,40,39,37,0,267,269,270,409,291],
    'lip_lower': [61,146,91,181,84,17,314,405,321,375,291],
}


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def contour_middle(mesh, indices):
    import numpy as np
    points=mesh[indices,:2]
    axis=points[-1]-points[0]
    length2=float(axis@axis)
    if length2<1:
        raise ValueError('Contour too small')
    t=(points-points[0])@axis/length2
    for i in range(len(points)-1):
        if (t[i]-.5)*(t[i+1]-.5)<=0 and abs(t[i+1]-t[i])>1e-8:
            alpha=(.5-t[i])/(t[i+1]-t[i])
            return points[i]+alpha*(points[i+1]-points[i])
    raise ValueError('No central contour intersection')


def mesh_points(mesh):
    result={name:mesh[index,:2].copy() for name,index in LANDMARKS.items()}
    for name,indices in CONTOURS.items():
        result[name]=contour_middle(mesh,indices)
    return result


def infer_variants(detector,image):
    import cv2
    import mediapipe as mp
    import numpy as np
    height,width=image.shape[:2]
    variants=[]
    for margin,maximum in [(0,0),(.08,896),(.16,1024)]:
        border=round(max(height,width)*margin)
        padded=cv2.copyMakeBorder(image,border,border,border,border,cv2.BORDER_CONSTANT,value=(127,127,127))
        scale=maximum/max(padded.shape[:2]) if maximum else 1
        scaled=cv2.resize(padded,(round(padded.shape[1]*scale),round(padded.shape[0]*scale))) if maximum else padded
        sx=scaled.shape[1]/padded.shape[1];sy=scaled.shape[0]/padded.shape[0]
        result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(cv2.cvtColor(scaled,cv2.COLOR_BGR2RGB))))
        converted=[]
        for landmarks in result.face_landmarks:
            mesh=np.asarray([[p.x*scaled.shape[1]/sx-border,p.y*scaled.shape[0]/sy-border] for p in landmarks])
            converted.append(mesh_points(mesh))
        variants.append(converted)
    if not variants[0]:
        raise ValueError('No face found in original image; old annotations are retained')
    predictions=[]
    for face in variants[0]:
        eye_distance=float(np.linalg.norm(face['left_eye_outer']-face['right_eye_outer']))
        if eye_distance<12:
            raise ValueError('Face too small for reliable refinement')
        candidates=[face]
        for variant in variants[1:]:
            if not variant:
                continue
            other=min(variant,key=lambda v:float(np.linalg.norm(v['nose_tip']-face['nose_tip'])))
            if float(np.linalg.norm(other['nose_tip']-face['nose_tip']))<eye_distance*.3:
                candidates.append(other)
        if len(candidates)<2:
            raise ValueError('Face was not stable across local inference variants')
        points={};spread=[]
        for name in LANDMARKS:
            samples=np.asarray([c[name] for c in candidates])
            median=np.median(samples,axis=0)
            spread.append(float(np.max(np.linalg.norm(samples-median,axis=1)))/eye_distance)
            x,y=median
            if not math.isfinite(x) or not math.isfinite(y) or not (0<=x<width and 0<=y<height):
                raise ValueError('A landmark is outside the image')
            points[name]=[round(float(x),2),round(float(y),2)]
        if max(spread)>.18:
            raise ValueError('Unstable landmark consensus; old annotations are retained')
        predictions.append((points,{'variants':len(candidates),'max_normalized_spread':max(spread)}))
    return predictions


def scope():
    job=request('/api/jobs/5')
    if job['id']!=5 or job['task_id']!=5 or job['type']!='annotation' or job['mode']!='annotation' or job.get('replicas_count',0):
        raise ValueError('Unexpected job scope')
    labels=request('/api/labels',params={'job_id':5,'page_size':100})
    parent,children=resolve_labels(labels)
    meta=request('/api/jobs/5/data/meta')
    return job,labels,parent,children,meta,build_frames(job,meta)


def prepare():
    import cv2
    import numpy as np
    job,labels,parent,children,meta,frames=scope()
    before=request('/api/jobs/5/annotations')
    if any(t['label_id']==parent['id'] for t in before.get('tracks',[])):
        raise ValueError('Cannot replace face tracks with per-image shapes')
    previous=[s for s in before['shapes'] if s['label_id']==parent['id']]
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=FOLDER/f'redo_{stamp}'
    folder.mkdir(parents=True)
    write(folder/'original_annotations.json',before)
    write(folder/'labels.json',labels);write(folder/'meta.json',meta);write(folder/'job.json',job)
    detector=create_detector()
    plan={'job_id':5,'task_id':5,'model_sha256':MODEL_SHA,
        'original_face_ids':[s['id'] for s in previous],'frames':[],'shapes':[],
        'method':'Three local inference variants; contour intersections at eye/mouth midline'}
    changed=0
    try:
        for number,info in frames.items():
            raw=request('/api/jobs/5/data',params={'type':'frame','number':number,'quality':'original'},binary=True)
            image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None or image.shape[:2]!=(info.height,info.width):
                raise ValueError('Image dimensions do not match metadata')
            predictions=infer_variants(detector,image)
            old=[s for s in previous if s['frame']==number]
            if len(predictions)<len(old):
                raise ValueError('New draft detects fewer faces than existing annotations')
            new=[build_shape(number,points,parent,children) for points,quality in predictions]
            plan['shapes'].extend(new)
            plan['frames'].append({'number':number,'quality':[q for p,q in predictions]})
            for shape in new:
                if old:
                    old_points={e['label_id']:e['points'] for e in old[0]['elements']}
                    changed+=sum(e['label_id'] not in old_points or math.dist(e['points'],old_points[e['label_id']])>1 for e in shape['elements'])
            preview=folder/'review';preview.mkdir(exist_ok=True)
            make_preview(image.copy(),new,children,preview/f'{number:06d}.jpg')
        write(folder/'plan.json',plan)
        write(folder/'status.json',{'state':'prepared','old_faces':len(previous),'new_faces':len(plan['shapes']),
            'frames':len(frames),'changed_keypoints':changed,'review_required':True})
        write(POINTER,{'folder':str(folder)})
        print('REPLACEMENT_PREPARED',json.dumps({'frames':len(frames),'old_faces':len(previous),
            'new_faces':len(plan['shapes']),'changed_keypoints':changed}),flush=True)
    finally:
        detector.close()


def apply():
    folder=Path(json.loads(POINTER.read_text(encoding='utf-8'))['folder']).resolve()
    if not folder.is_relative_to(FOLDER.resolve()):
        raise ValueError('Invalid backup location')
    plan=json.loads((folder/'plan.json').read_text(encoding='utf-8'))
    baseline=json.loads((folder/'original_annotations.json').read_text(encoding='utf-8'))
    job,labels,parent,children,meta,frames=scope()
    if plan['job_id']!=5 or plan['task_id']!=5 or labels!=json.loads((folder/'labels.json').read_text(encoding='utf-8')) or meta!=json.loads((folder/'meta.json').read_text(encoding='utf-8')):
        raise ValueError('Schema/media changed since preparation')
    old_ids=set(plan['original_face_ids'])
    actual_old_ids={s['id'] for s in baseline['shapes'] if s['label_id']==parent['id']}
    if old_ids!=actual_old_ids:
        raise ValueError('Deletion scope does not match the backed-up face labels')
    required={l['id'] for l in children.values()}
    if {s['frame'] for s in plan['shapes']}!=set(frames):
        raise ValueError('Replacement does not cover the complete job')
    for shape in plan['shapes']:
        if shape.get('id') is not None or shape['type']!='skeleton' or shape['label_id']!=parent['id'] or len(shape['elements'])!=13 or {e['label_id'] for e in shape['elements']}!=required:
            raise ValueError('Invalid replacement skeleton')
        info=frames[shape['frame']]
        for e in shape['elements']:
            x,y=e['points']
            if e.get('id') is not None or e['frame']!=shape['frame'] or e['type']!='points' or not (math.isfinite(x) and math.isfinite(y) and 0<=x<info.width and 0<=y<info.height):
                raise ValueError('Invalid replacement keypoint')
    remaining_baseline=copy.deepcopy(baseline)
    remaining_baseline['shapes']=[s for s in baseline['shapes'] if s['id'] not in old_ids]
    live=request('/api/jobs/5/annotations')
    preserve(remaining_baseline,live)
    baseline_ids={s['id'] for s in baseline['shapes']}
    def replacements(data):
        return [s for s in data['shapes'] if s['id'] not in baseline_ids]
    current_old={s['id'] for s in live['shapes']} & old_ids
    if current_old and current_old!=old_ids:
        raise ValueError('Only some old faces remain; review the saved backup before proceeding')
    if current_old:
        preserve(baseline,live)
    fresh=replacements(live)
    planned={signature(s) for s in plan['shapes']}
    if any(signature(s) not in planned for s in fresh):
        raise ValueError('Unreviewed annotations appeared during replacement')
    known={signature(s) for s in fresh}
    pending=[s for s in plan['shapes'] if signature(s) not in known]
    if not current_old and pending:
        raise ValueError('Old labels are absent but replacement is incomplete; stop')
    while pending:
        shape=pending.pop(0)
        write(folder/'status.json',{'state':'creating_replacements','remaining':len(pending)+1})
        request('/api/jobs/5/annotations',method='PATCH',params={'action':'create'},
            payload={'version':live['version'],'shapes':[shape],'tracks':[],'tags':[]})
        after=request('/api/jobs/5/annotations')
        preserve(live,after)
        old_live_ids={s['id'] for s in live['shapes']}
        created=[s for s in after['shapes'] if s['id'] not in old_live_ids]
        if len(created)!=1:
            raise ValueError('Unexpected create result; do not retry blindly')
        verify_predictions([shape],created)
        live=after
        write(folder/'after_create.json',live)
    fresh=replacements(live)
    if len(fresh)!=len(plan['shapes']):
        raise ValueError('Replacement count is inconsistent; old annotations retained')
    verify_predictions(plan['shapes'],fresh)
    if current_old:
        current=request('/api/jobs/5/annotations')
        preserve(live,current)
        if {s['id'] for s in current['shapes']}!={s['id'] for s in live['shapes']}:
            raise ValueError('Annotations changed immediately before delete')
        write(folder/'before_delete.json',current)
        delete_shapes=[s for s in current['shapes'] if s['id'] in old_ids]
        write(folder/'status.json',{'state':'deleting_original_faces','old_ids':sorted(old_ids)})
        request('/api/jobs/5/annotations',method='PATCH',params={'action':'delete'},
            payload={'version':current['version'],'shapes':delete_shapes,'tags':[],'tracks':[],'intervals':[]})
        live=request('/api/jobs/5/annotations')
    preserve(remaining_baseline,live)
    verify_predictions(plan['shapes'],replacements(live))
    if any(s['id'] in old_ids for s in live['shapes']):
        raise ValueError('Old face annotations remain after replacement')
    expected_ids={s['id'] for s in remaining_baseline['shapes']} | {s['id'] for s in fresh}
    if {s['id'] for s in live['shapes']}!=expected_ids:
        raise ValueError('Final annotation IDs differ from the verified scope')
    write(folder/'verified_annotations.json',live)
    summary={'state':'verified','frames':len(frames),'old_faces_removed':len(old_ids),
        'new_faces_verified':len(fresh),'keypoints':sum(len(s['elements']) for s in fresh),
        'other_annotations_preserved':True,'review_required':True}
    write(folder/'status.json',summary)
    write(FOLDER/'redo_apply_status.json',summary)
    print('REPLACEMENT_VERIFIED',json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    try:
        apply() if args.apply else prepare()
    except Exception as error:
        write(FOLDER/'redo_error_private.json',{'type':type(error).__name__,'detail':str(error)})
        print('REPLACEMENT_STOPPED',type(error).__name__,'Private details saved locally.',flush=True)
        raise SystemExit(1)
