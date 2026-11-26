"""Build the visually reviewed Day12-G07 cabin annotation, using live label IDs.

No model prediction, reference ground truth, credentials, or external service.
The user explicitly requested annotating children and occupied seats despite
the lab's empty-seat-only assumption. This exception is recorded in review notes.
"""
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageEnhance

ROOT = Path(__file__).resolve().parent / 'outputs' / 'cabin'


def main():
    snap = json.loads((ROOT / 'live_snapshot.json').read_text(encoding='utf-8'))
    assert (snap['job']['id'], snap['job']['task_id'], snap['job']['task_name']) == (11, 11, 'Day12-G07')
    assert not any(snap['annotations'][k] for k in ('tags', 'shapes', 'tracks'))
    labels = {x['name']: x for x in snap['labels']['results']}
    frames = snap['meta']['frames']
    payload = {'version': snap['annotations']['version'], 'tags': [], 'shapes': [], 'tracks': []}

    def attrs(label, values):
        specs = labels[label]['attributes']
        assert set(values) <= {a['name'] for a in specs}
        result = []
        for a in specs:
            value = values.get(a['name'], a['default_value'])
            assert value in a['values'], (label, a['name'], value)
            result.append({'spec_id': a['id'], 'value': value})
        return result

    def box(frame, label, xyxy, group=0, occluded=False, **values):
        x1, y1, x2, y2 = xyxy
        meta = frames[frame]
        assert 0 <= x1 < x2 <= meta['width'] and 0 <= y1 < y2 <= meta['height']
        payload['shapes'].append(dict(type='rectangle', frame=frame, label_id=labels[label]['id'],
                                      points=xyxy, occluded=occluded, outside=False,
                                      rotation=0, z_order=0, group=group, source='manual',
                                      attributes=attrs(label, values)))

    for f, meta in enumerate(frames):
        payload['tags'].append(dict(frame=f, label_id=labels['frame_meta']['id'], group=0,
                                    source='manual', attributes=attrs('frame_meta', {
                                        'modality': 'rgb' if '_rgb' in meta['name'] else 'ir',
                                        'light': 'unknown', 'camera_blocked': 'no'})))

    # Original, unrotated image coordinates. Left/right are anatomical.
    drivers = [
        ([223, 346, 1212, 860], [845, 365, 1140, 671], 'worn_correct',
         [([456, 389, 595, 495], 'right'), ([411, 512, 571, 744], 'left')]),
        ([0, 232, 1268, 899], [905, 315, 1204, 597], 'worn_correct', []),
        ([227, 415, 1278, 908], [952, 442, 1232, 710], 'worn_correct',
         [([408, 431, 577, 524], 'right'), ([786, 477, 1000, 682], 'left')]),
        ([224, 382, 1251, 884], [912, 440, 1175, 721], 'worn_correct',
         [([431, 468, 546, 561], 'right'), ([471, 548, 580, 661], 'left')]),
        ([255, 367, 1207, 882], [862, 384, 1146, 653], 'worn_correct',
         [([781, 371, 949, 493], 'right')]),
        ([223, 417, 1279, 974], [928, 508, 1220, 758], 'unknown',
         [([330, 798, 574, 971], 'left')]),
    ]
    # Point order follows the live skeleton schema. None means unlocalizable,
    # represented as outside, not an invented hidden landmark.
    landmarks = [
        [None]*8 + [(914, 420), (887, 497), (890, 451), (878, 474), (865, 479)],
        [None]*8 + [(1004, 373), (973, 455), (979, 394), (966, 424), (953, 424)],
        [(1113, 613), (1094, 555), (1113, 584), (1097, 587),
         (1074, 473), (1084, 511), (1091, 491), (1078, 492)] + [None]*5,
        [None]*8 + [(970, 488), (956, 589), (949, 533), (950, 562), (937, 562)],
        [(1041, 567), (1031, 514), (1047, 540), (1031, 541),
         (1006, 410), (1014, 450), (1021, 430), (1007, 432),
         (945, 479), (950, 575), (952, 504), (947, 542), (931, 546)],
        [None]*8 + [(1005, 697), (1000, 723), None, (987, 710), (975, 710)],
    ]
    for f, (body, face, belt, hands) in enumerate(drivers):
        group = f + 1
        box(f, 'person', body, group=group, seat='driver', seatbelt=belt,
            occluded=f in (1, 5))
        visible_eyes = f in (2, 4)
        box(f, 'face', face, group=group, occluded=f in (2, 5),
            eye_left='open' if visible_eyes else 'unknown',
            eye_right='open' if visible_eyes else 'unknown',
            glasses='none' if visible_eyes else 'unknown', gaze_zone='other')
        elements = []
        for spec, pt in zip(labels['face13']['sublabels'], landmarks[f], strict=True):
            elements.append(dict(type='points', frame=f, label_id=spec['id'],
                                 points=list(pt) if pt else [0, 0],
                                 outside=pt is None, occluded=pt is None,
                                 rotation=0, z_order=0, group=group, source='manual', attributes=[]))
        payload['shapes'].append(dict(type='skeleton', frame=f, label_id=labels['face13']['id'],
                                      points=[], rotation=0, occluded=f in (0, 1, 2, 3, 5),
                                      outside=False, z_order=0, group=group, source='manual',
                                      attributes=[], elements=elements))
        for xyxy, side in hands:
            box(f, 'hand', xyxy, group=group, side=side, whose='driver', on_wheel='no')

    # Geometry checked on RGB and IR separately; shared shape positions are
    # appropriate here because each supplied pair has identical camera pose.
    for f in range(6, 22):
        scene = (f - 6) // 2 + 1
        group = 100 + scene
        if scene == 1:
            box(f, 'child_seat', [116, 201, 381, 628], seat='rear_right', occupied='no', occluded=True)
            box(f, 'child_seat', [299, 164, 640, 639], seat='rear_middle', occupied='yes', occluded=True)
            box(f, 'person', [355, 231, 578, 524], group=group, seat='rear_middle', seatbelt='not_worn')
            box(f, 'face', [425, 274, 496, 333], group=group)
        elif scene == 2:
            box(f, 'infant_seat', [0, 323, 386, 639], seat='rear_right', facing='rear')
            box(f, 'person', [558, 97, 879, 640], group=group, seat='rear_left', seatbelt='not_worn')
            # Face turned away; neither eyes nor nose are visible.
        elif scene == 3:
            box(f, 'object', [191, 316, 372, 544], seat='rear_right', type='bag')
            box(f, 'infant_seat', [345, 318, 628, 627], seat='rear_middle', facing='rear', occluded=True)
            box(f, 'person', [416, 363, 531, 541], group=group, seat='rear_middle', seatbelt='unknown')
            box(f, 'object', [598, 398, 646, 450], seat='rear_left', type='cup')
        elif scene in (4, 6):
            box(f, 'infant_seat', [0, 321, 382, 639], seat='rear_right', facing='rear')
            box(f, 'child_seat', [559, 164, 933, 639], seat='rear_left', occupied='no')
        elif scene == 5:
            box(f, 'infant_seat', [0, 320, 382, 627], seat='rear_right', facing='rear', occluded=True)
            box(f, 'person', [145, 367, 334, 538], group=group, seat='rear_right', seatbelt='unknown')
        elif scene == 7:
            box(f, 'object', [210, 352, 378, 591], seat='rear_right', type='box')
            box(f, 'child_seat', [364, 204, 591, 624], seat='rear_middle', occupied='no')
        elif scene == 8:
            box(f, 'child_seat', [117, 201, 407, 627], seat='rear_right', occupied='yes', occluded=True)
            box(f, 'person', [91, 200, 384, 538], group=group, seat='rear_right', seatbelt='not_worn')
            box(f, 'face', [273, 215, 348, 295], group=group)
            box(f, 'person', [526, 77, 960, 640], group=group+100, seat='rear_left', seatbelt='not_worn')
            box(f, 'face', [670, 111, 768, 219], group=group+100)

    (ROOT / 'plan.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    export_xml(payload, snap, ROOT / 'annotations_draft.xml')
    render(payload, snap)
    notes = '''Day12-G07 / Task 11 / Job 11 — review notes

User override (2026-11-21): annotate children as person and seats as actually
shown. S01/S08 child seats are occupied=yes, which the LIVE schema allows.
The classroom guideline's assumption that all seats are empty is contradicted
by these images. This intentional exception produces 4 cabin_check errors.
No project labels or attribute definitions were changed.

All left/right seats are vehicle-relative (image left = rear_right).
All DMS left/right hands/eyes are person-relative; original images unrotated.
Light is unknown: rendered OMS images and NIR brightness do not establish time.
All 22 frames have one frame_meta tag. No external image inference service used.

D01/D02/D04: eye details behind NIR glasses/reflections are not reliable;
eye states and glasses type unknown, unlocalizable eye landmarks outside.
D03: drinking bottle, not phone/object. Mouth and nose hidden by bottle/hand;
their landmarks outside. Two visible hands; neither touching wheel.
D02: no visible hand (extended arm is not a hand). No hand box fabricated.
D05: visible raised right hand; no clearly recognizable phone edge/corner;
no phone box inferred merely from hand at ear. Recheck this ambiguous region.
D06: right arm obscures right eye and right hand; only left hand is localized.
Seatbelt visibility insufficient to establish status, hence unknown.
D01-D06 gaze_zone=other: precise target not established from single NIR frame.
Landmarks are manually localized; hidden points marked outside, not guessed.

OMS face attributes are schema defaults only; guideline requires face box only.
S02: adult turned away, no visible eyes/nose; no face box.
S03/S05: infant face turned away/hidden; no face box.
S03: backpack rear_right; small cup rear_left; infant and carrier rear_middle.
SVIRO seatbelt never worn_correct/worn_incorrect. Children in carriers have
unknown where torso is not sufficient to assess a belt.

Empty rear-seat counts per scene: S01=1, S02=1, S03=0, S04=1,
S05=2, S06=1, S07=1, S08=1. Seats containing objects count as occupied.

This is a reviewed annotation draft. A second human quality review remains
appropriate for manual landmark precision, ambiguous NIR details, and the
intentional classroom-guideline exception; Job state is not auto-completed.
'''
    (ROOT / 'review_notes.txt').write_text(notes, encoding='utf-8')
    print('Tags:', len(payload['tags']), 'Shapes:', len(payload['shapes']))
    print(Counter(next(l['name'] for l in labels.values() if l['id']==s['label_id']) for s in payload['shapes']))


def export_xml(data, snap, destination):
    labels = {x['id']: x for x in snap['labels']['results']}
    all_labels = dict(labels)
    for label in labels.values():
        all_labels.update({x['id']: x for x in label.get('sublabels', [])})
    root = ET.Element('annotations')
    ET.SubElement(root, 'version').text = '1.1'
    for f, meta in enumerate(snap['meta']['frames']):
        img = ET.SubElement(root, 'image', id=str(f), name=meta['name'], width=str(meta['width']), height=str(meta['height']))
        for shape in data['tags'] + data['shapes']:
            if shape['frame'] != f:
                continue
            label = all_labels[shape['label_id']]
            common = {'label': label['name'], 'source': shape.get('source', 'manual')}
            kind = shape.get('type')
            if kind == 'rectangle':
                common.update({k: str(v) for k, v in zip(('xtl','ytl','xbr','ybr'), shape['points'])})
                common['occluded'] = str(int(shape.get('occluded', False)))
                element = ET.SubElement(img, 'box', common)
            elif kind == 'skeleton':
                element = ET.SubElement(img, 'skeleton', common)
                for point in shape['elements']:
                    ET.SubElement(element, 'points', label=all_labels[point['label_id']]['name'],
                                  points=','.join(str(v) for v in point['points']),
                                  occluded=str(int(point.get('occluded', False))),
                                  outside=str(int(point.get('outside', False))))
            elif kind is None:
                element = ET.SubElement(img, 'tag', common)
            else:
                raise ValueError(kind)
            spec_names = {x['id']:x['name'] for x in label.get('attributes', [])}
            for attr in shape['attributes']:
                ET.SubElement(element, 'attribute', name=spec_names[attr['spec_id']]).text = attr['value']
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding='utf-8', xml_declaration=True)


def render(data, snap):
    labels = {x['id']: x for x in snap['labels']['results']}
    dest = ROOT / 'review' / 'annotated'
    dest.mkdir(exist_ok=True)
    for f, meta in enumerate(snap['meta']['frames']):
        im = Image.open(ROOT / 'frames' / meta['name']).convert('RGB')
        im = ImageEnhance.Brightness(im).enhance(1.7 if f < 6 else 1.25)
        draw = ImageDraw.Draw(im)
        for shape in data['shapes']:
            if shape['frame'] != f:
                continue
            label = labels[shape['label_id']]
            color = label['color']
            if shape['type'] == 'rectangle':
                draw.rectangle(shape['points'], outline=color, width=2)
                attr = {a['spec_id']:a['value'] for a in shape['attributes']}
                suffix = ' '.join(attr[a['id']] for a in label['attributes'] if a['name'] in ('seat','side','occupied'))
                draw.text(tuple(shape['points'][:2]), label['name']+' '+suffix, fill=color, stroke_width=1, stroke_fill='black')
            else:
                for p in shape['elements']:
                    if p['outside']:
                        continue
                    x,y=p['points']
                    draw.ellipse((x-2,y-2,x+2,y+2), fill='#00ff00')
        im.save(dest / meta['name'])


if __name__ == '__main__':
    main()
