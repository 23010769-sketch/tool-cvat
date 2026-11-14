"""Image-reviewed corrections for the local Job 4 draft; no server writes."""
import copy
import json
from collections import defaultdict

import cv2

from annotate_local_pose import EDGES, ROOT, save

folder = ROOT / 'job4'
plan = json.loads((folder / 'plan.json').read_text(encoding='utf-8'))
save(folder / 'plan_before_refinement.json', plan)
frames = defaultdict(list)
for shape in plan['shapes']:
    frames[shape['frame']].append(shape)


def change(frame, person, index, *, xy=None, outside=None, occluded=None):
    point = frames[frame][person]['elements'][index]
    if xy is not None:
        point['points'] = xy
        point['outside'] = False
    if outside is not None:
        point['outside'] = outside
    if occluded is not None:
        point['occluded'] = occluded
    point['source'] = 'manual'


# Near tennis player: far eye is hidden; the near eye and ear are visible.
change(1, 0, 1, xy=[280, 130], occluded=True)
change(1, 0, 3, outside=True)
# Rear tennis official: most of the face is above the image, the far limbs
# are obscured by the foreground player.
change(1, 1, 0, xy=[331, 7])
for index in [1, 2, 3, 4]:
    change(1, 1, index, outside=True)
change(1, 1, 9, xy=[301, 211])
for index in [10, 12, 14, 16]:
    change(1, 1, index, occluded=True)
# Girl's lower legs extend below the image; do not put ankles on the knees.
for index in [15, 16]:
    change(6, 0, index, outside=True)
# Station balcony: blurred silhouettes and a solid railing conceal the
# face details, hands, pelvis and legs. Keep only supported shoulder/arm points.
for person in [0, 1]:
    for index in list(range(5)) + [9, 10, 11, 12, 13, 14, 15, 16]:
        change(7, person, index, outside=True)
# Cake scene: the man's left arm lies behind the couple; lower legs are
# hidden by the table / image boundary. The woman's knees are concealed.
for index in [5, 7, 9, 11]:
    change(8, 0, index, occluded=True)
for index in [13, 14, 15, 16]:
    change(8, 0, index, outside=True)
for index in [13, 14, 15, 16]:
    change(8, 1, index, outside=True)
for index in [3, 4]:
    change(8, 2, index, occluded=True)
for index in [13, 14, 15, 16]:
    change(8, 2, index, outside=True)
# Skateboarder's wrists / ankles adjusted to the visible joints.
change(11, 0, 10, xy=[56, 78])
change(11, 0, 9, xy=[349, 143])
change(11, 0, 15, xy=[279, 521])
change(11, 0, 16, xy=[181, 435])

plan['review_notes'] = {
    '7': 'Very blurred distant people: no reliable face/hand/lower-body landmarks; these keypoints are outside/unlabeled. Additional tiny background silhouettes are not resolved.',
    '8': 'Cake table and other people hide lower-body landmarks. Small background faces are not resolved; review scope if they must be included.',
    'all': 'Model predictions plus visual corrections, with 17 elements per skeleton. Existing manual skeleton is preserved byte-for-byte. Job stage/state is not changed.'
}
save(folder / 'plan.json', plan)
existing = json.loads((folder / 'annotations.json').read_text(encoding='utf-8'))
for frame in range(12):
    image = cv2.imread(str(folder / 'images' / f'{frame:06d}.jpg'))
    shapes = [s for s in existing['shapes'] if s['frame']==frame] + frames[frame]
    for shape in shapes:
        elements = sorted(shape['elements'], key=lambda e:e['label_id'])
        for a,b in EDGES:
            if not elements[a]['outside'] and not elements[b]['outside']:
                color = (0,180,255) if elements[a]['occluded'] or elements[b]['occluded'] else (50,230,50)
                cv2.line(image, tuple(map(round,elements[a]['points'])), tuple(map(round,elements[b]['points'])), color, 2)
        for index,element in enumerate(elements):
            if element['outside']:
                continue
            point=tuple(map(round,element['points']))
            color=(0,180,255) if element['occluded'] else (50,230,50)
            cv2.circle(image,point,3,color,-1)
    cv2.putText(image,f'Frame {frame} | {len(shapes)} skeletons',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
    cv2.imwrite(str(folder / 'review' / f'{frame:06d}.jpg'),image)
save(folder / 'review_notes.json', plan['review_notes'])
print('Refined',len(plan['shapes']),'new skeletons. Existing manual skeleton unchanged.')
