"""Image-reviewed corrections for the user's Job 206, without ground-truth data."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent / 'outputs' / 'job206_guideline'
baseline = json.loads((ROOT / 'snapshot_before.json').read_text(encoding='utf-8'))
labels = {x['name']: x for x in baseline['labels']['results']}
names = {x['id']: x['name'] for x in labels.values()}
objects = {'pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle', 'traffic light', 'traffic sign'}
allowed = objects | {'area/direct', 'area/alternative', 'lane/crosswalk',
    'lane/double white', 'lane/double yellow', 'lane/double other', 'lane/road curb',
    'lane/single white', 'lane/single yellow', 'lane/single other'}
old = {x['id']: copy.deepcopy(x) for x in baseline['annotations']['shapes']}
kept = {key: copy.deepcopy(value) for key, value in old.items() if names[value['label_id']] in allowed}
created = []
issues = []
reasons = {}


def attributes(name, occluded=False, truncated=False):
    values = {'occluded': occluded, 'truncated': truncated}
    return [{'spec_id': x['id'], 'value': str(values.get(x['name'], False)).lower()}
            for x in labels[name]['attributes']]


def shape(f, name, kind, pts, occ=False, trunc=False):
    created.append({'frame': f, 'label_id': labels[name]['id'], 'type': kind,
        'points': pts, 'group': 0, 'source': 'manual', 'occluded': occ,
        'outside': False, 'z_order': 0, 'rotation': 0,
        'attributes': attributes(name, occ, trunc), 'elements': []})


def box(f, name, pts, occ=False, trunc=False):
    shape(f, name, 'rectangle', pts, occ, trunc)


def polygon(f, name, pts):
    shape(f, 'area/' + name, 'polygon', pts)


def line(f, name, pts):
    shape(f, 'lane/' + name, 'polyline', pts)


def drop(ids, reason):
    for key in ids:
        kept.pop(key)
        reasons[str(key)] = reason


def edit(key, *, pts=None, name=None, occ=None, trunc=None):
    item = kept[key]
    original_name = names[item['label_id']]
    original_attrs = {x['name']: x['id'] for x in labels[original_name]['attributes']}
    original_values = {x['spec_id']: x['value'] for x in item['attributes']}
    if occ is None:
        occ = item['occluded'] or original_values.get(original_attrs.get('occluded')) == 'true'
    if trunc is None:
        trunc = original_values.get(original_attrs.get('truncated')) == 'true'
    if name is None: name = original_name
    item.update(label_id=labels[name]['id'], source='manual', occluded=occ,
                attributes=attributes(name, occ, trunc))
    if pts is not None: item['points'] = pts


def issue(f, kind, pts, message):
    issues.append({'frame': f, 'position': pts, 'message': kind + ': ' + message})


# Review of detection rectangles against the actual pixels, before drawing areas.
drop([79975, 80242], 'Reflection/ego bonnet is not a separate car')
drop([86959, 86960, 80186, 80183, 80185, 87279, 80212, 87306, 80220,
      80225, 80236, 80237, 80243, 87381], 'Duplicate or conflicting label on the same object')
drop([87213], 'Airline/terminal commercial sign, outside traffic-sign scope')
drop([87239], 'Building/storefront region, not a car')
drop([80213], 'Fire hydrant, not a pedestrian')
drop([80233], 'Green windshield reflection, not a bus')
drop([80232], 'Wheeled food-vendor cart, outside listed vehicle classes; issue for mentor')

# Apply border truncation to object boxes, then explicitly review occlusions below.
for key, item in kept.items():
    p = item['points']
    if item['type'] == 'rectangle':
        edit(key, trunc=min(p[0], p[1]) < 2 or p[2] > 1278 or p[3] > 718)

for key in [79021, 86885, 79971, 86934, 86933, 86941, 79979, 79980,
            86961, 86962, 79984, 79985, 79986, 79987, 87221, 87222,
            87223, 87225, 80184, 87226, 87227, 87228, 80194,
            80198, 87276, 80211, 80208, 87278, 87277, 87295, 87296,
            87293, 87290, 80218, 80217, 87307, 87328, 80226, 80227,
            80228, 80229, 80230, 80231, 87330, 87331, 87332, 87333,
            87334, 87329, 87342, 87343, 87366, 87368, 87369,
            87379, 80239, 80241, 87384, 87383, 87382]:
    edit(key, occ=True)

# Frame 80: bicycle dock, stop light, and one travel lane.
edit(79019, pts=[104,129,578,512])
edit(79018, pts=[0,250,116,366], trunc=True)
edit(79021, pts=[555,245,608,289], occ=True)
edit(86885, pts=[929,243,1172,475], occ=True)
edit(86884, pts=[1005,240,1279,517], occ=True, trunc=True)
box(80, 'bicycle', [835,240,1082,426], True)
box(80, 'bicycle', [875,242,1123,450], True)
box(80, 'bicycle', [1197,246,1279,614], True, True)
for p in [[78,190,99,234], [616,154,634,184], [620,195,637,221],
          [768,0,788,37], [821,0,846,23]]: box(80,'traffic light',p,trunc=p[1]==0)
polygon(80,'direct',[0,383,105,353,111,432,127,474,173,503,279,519,
    530,517,573,498,578,460,577,319,616,316,960,546,804,542,610,535,
    409,535,203,545,0,563])
line(80,'single yellow',[0,381,107,353])
line(80,'single white',[776,416,844,460,915,507,977,548])
line(80,'single white',[594,319,619,335])
line(80,'road curb',[753,305,783,312,809,323,834,335])
issue(80,'LOW_VISIBILITY',[970,235,1279,620],
      'Cum xe dap ben phai che khuat lan nhau; can reviewer xac nhan so instance va bien xe ngoai cung.')

# Frame 81: dark street, parked cars and a separate left foreground vehicle.
polygon(81,'direct',[430,516,590,487,647,480,686,475,692,503,805,504,
    808,485,862,488,868,507,940,536,1037,582,1080,642,851,618,587,616,304,652,250,634])
issue(81,'UNCERTAIN_BOUNDARY',[270,465,1120,650],
      'Mat duong thieu sang, khong du bang chung xac dinh ranh lan va alternative; da khoanh phan direct nhin thay.')

# Frame 82: exclude the bonnet reflection; sidewalk people and real traffic lights.
for p in [[71,127,99,194], [128,190,143,224], [161,194,181,235],
          [292,233,308,264]]: box(82,'traffic light',p)
edit(79974, trunc=True)
polygon(82,'direct',[107,548,176,479,232,420,280,350,307,292,313,403,
    325,441,606,445,713,548])
polygon(82,'alternative',[620,445,705,443,720,389,809,398,954,391,
    1101,411,1104,505,1195,546,730,548])
line(82,'double yellow',[111,546,176,477,234,410,280,342,305,290])
line(82,'single white',[620,447,657,487,707,541])
issue(82,'LOW_VISIBILITY',[0,233,1134,371],
      'Xe va nguoi o xa/ben via he toi; reviewer kiem tra instance nho va occlusion.')

# Frame 83: access to street from a parking area; parked minivans stay separate.
edit(79979, occ=True)
edit(79980, occ=True)
box(83,'car',[725,207,876,266],True)
box(83,'car',[39,240,61,263])
polygon(83,'direct',[472,424,674,425,674,536,1038,540,1078,609,
    975,659,270,656,235,528,450,502])
line(83,'road curb',[470,425,674,425])
issue(83,'UNCERTAIN_SCOPE',[461,352,1064,663],
      'Anh o loi ra bai do xe, khong co ranh lan ro; can mentor chot pham vi drivable area tai loi ra.')

# Frame 84: two same-direction lanes; never join across broken paint gaps.
edit(79982, pts=[1213,314,1279,464],trunc=True)
edit(79983, pts=[1091,321,1229,422])
polygon(84,'direct',[192,608,349,544,482,488,594,436,677,399,704,377,
    716,347,743,350,790,371,816,393,843,427,899,482,974,548,1073,596,
    926,589,739,581,549,578,344,589])
polygon(84,'alternative',[743,350,774,351,777,356,814,357,817,362,860,368,
    865,374,913,377,943,386,997,395,1039,408,1091,424,1214,460,1279,489,
    1279,613,1073,596,974,548,899,482,843,427,816,393,790,371])
for p in [[904,455,928,480], [860,407,870,417], [835,382,842,389],
          [818,368,823,373]]: line(84,'single white',p)
line(84,'single white',[877,373,951,399,1050,431,1155,464,1279,490])
line(84,'road curb',[191,608,355,544,489,484,599,434,679,398,704,376,714,349])

# Frame 85: airport through lane; curbside loading space is not alternative.
edit(87214,name='bus',pts=[328,319,394,378],occ=True)
for p in [[31,247,51,270], [264,310,280,330], [339,267,360,287],
          [426,273,447,294], [486,317,505,339]]:box(85,'traffic light',p)
polygon(85,'direct',[59,644,124,583,188,513,237,448,273,394,290,378,
    294,435,398,437,400,400,420,400,435,450,469,450,468,474,548,507,
    649,563,762,635,710,646,552,645,303,646])
polygon(85,'alternative',[410,442,432,440,435,451,470,452,470,444,
    570,445,589,424,590,488,887,491,924,534,927,653,766,636,
    649,562,548,507,468,474])
line(85,'single yellow',[51,644,79,616])
line(85,'single yellow',[135,577,172,536])
line(85,'single yellow',[204,499,227,470])
line(85,'single yellow',[245,445,264,418])
line(85,'single yellow',[278,399,288,382])
line(85,'single white',[718,604,759,632])
line(85,'road curb',[0,620,73,567,143,501,205,431,252,380,284,350])

# Frame 86: pedestrians, parked cars, zebra crossing. Each stripe is a polyline.
edit(80180,pts=[171,225,268,392],occ=True)
edit(87221,pts=[514,221,610,303],occ=True)
box(86,'car',[342,187,367,214],True)
box(86,'car',[355,176,378,199],True)
box(86,'car',[368,164,389,183],True)
box(86,'car',[402,157,426,178])
polygon(86,'direct',[146,599,274,390,296,320,335,267,365,226,
    398,195,423,181,437,195,456,213,479,241,501,267,515,304,
    550,331,592,358,669,362,749,374,795,395,1005,430,1279,461,
    1279,611,1102,607,887,597,645,584,373,584])
for p in [[0,506,147,402], [299,519,373,394], [609,500,530,389],
          [922,484,695,383]]:line(86,'crosswalk',p)
line(86,'road curb',[748,347,765,365,799,389,864,405,958,424,1102,446,1279,460])
issue(86,'LOW_VISIBILITY',[245,148,532,278],
      'Nhieu xe do lien tiep o xa va phan dau nguoi bi vat trong xe che; can reviewer kiem tra completeness/occluded.')

# Frame 87: blurred traffic, curbside parking and two yellow center lines.
for p in [[598,370,616,392], [710,354,727,379], [732,378,749,402]]:box(87,'traffic light',p)
box(87,'car',[730,396,770,424],True)
polygon(87,'direct',[365,601,448,548,524,494,590,450,621,423,628,420,
    629,454,709,456,711,425,728,428,744,459,771,500,814,557,869,621,
    744,619,590,615])
polygon(87,'alternative',[710,454,723,429,750,425,767,437,768,475,
    893,477,932,490,1002,524,1116,570,1279,629,1279,651,1024,634,
    869,621,814,557,771,500,744,459])
line(87,'double yellow',[353,601,437,548,517,493,581,449])
for p in [[797,544,839,589], [745,480,766,503], [719,447,731,461]]:line(87,'single white',p)
line(87,'road curb',[902,473,953,491,1026,522,1110,557,1204,596,1279,625])
issue(87,'LOW_VISIBILITY',[196,348,1091,490],
      'Motion blur manh; can kiem tra bien xe, den tin hieu va nguoi tren via he. Khong tao instance tu bong mo.')

# Frame 88: backs of traffic signal heads and separate warning sign panels.
edit(80195,pts=[0,297,783,719],trunc=True)
edit(87246,pts=[204,196,308,300])
box(88,'traffic sign',[309,196,384,300])
for p in [[65,0,109,96], [111,0,152,94], [153,0,196,94]]:box(88,'traffic light',p,trunc=True)
polygon(88,'direct',[784,519,975,519,976,479,1027,481,1201,482,
    1217,503,1207,539,1214,580,1233,623,1279,686,1279,710,963,710,784,707])
line(88,'single white',[1279,506,1232,522,1211,552,1218,588,1239,628,1279,682])

# Frame 89: van is a passenger van; right same-direction lane is alternative.
polygon(89,'direct',[276,488,371,409,453,349,527,311,594,292,
    598,322,675,323,677,313,689,345,727,396,780,447,830,488])
polygon(89,'alternative',[679,310,737,298,783,295,780,310,809,337,
    870,360,969,389,1099,419,1279,458,1279,489,837,489,779,448,
    726,396,688,344])
line(89,'single yellow',[275,488,368,410,450,351,526,312,596,291])
line(89,'single white',[680,321,693,347,726,392,774,442,831,486])
line(89,'single white',[799,314,827,339,890,363,984,389,1108,421,1279,456])

# Frame 90: small real pedestrians ahead; no alternative into opposing traffic.
for p in [[326,312,343,358], [345,310,362,360]]:box(90,'pedestrian',p)
for p in [[343,304,375,337], [373,315,400,342], [402,320,424,344],
          [426,325,441,345]]:box(90,'car',p,True)
polygon(90,'direct',[425,584,429,531,437,476,444,425,448,387,453,347,
    510,348,570,354,579,375,616,387,654,397,657,455,960,457,
    1045,511,1090,552,1093,605,947,599,747,587])
for p in [[393,489,411,463], [421,440,430,417], [440,393,443,381],
          [449,367,451,357]]:line(90,'single white',p)
for p in [[310,375,331,371], [350,373,366,370], [385,372,402,371],
          [420,372,434,371], [455,374,478,374], [496,375,520,376],
          [537,378,561,379]]:line(90,'crosswalk',p)
issue(90,'UNCERTAIN_BOUNDARY',[300,337,659,493],
      'Nut giao/hep duong, vach huong xe mo; reviewer xac nhan ranh direct, khong suy dien alternative sang lan nguoc chieu.')

# Frame 91: rider is on a two-wheeler; hydrant must not be labelled as a person.
edit(80209,name='rider',pts=[536,326,567,386])
box(91,'motorcycle',[536,357,569,398])
box(91,'pedestrian',[704,316,721,354])
box(91,'pedestrian',[724,312,743,355],True)
box(91,'car',[120,348,168,373],True)
box(91,'car',[238,337,268,355],True)
box(91,'car',[269,335,295,351],True)
polygon(91,'direct',[313,600,329,504,343,444,357,395,369,362,
    399,354,399,361,438,361,436,354,464,356,468,364,512,365,
    528,377,534,398,569,400,655,413,771,440,887,468,1044,503,
    1202,540,1279,557,1279,653,1179,637,1039,614,899,605,
    731,582,560,574,418,580])
polygon(91,'alternative',[50,467,121,410,203,377,279,355,366,350,
    369,362,357,395,343,444,329,504,313,600,236,612,89,615,51,566])
for p in [[299,487,309,464], [328,413,333,401], [343,382,346,373],
          [351,366,353,361]]:line(91,'single white',p)
line(91,'road curb',[557,392,655,413,771,440,887,468,1044,503,1202,540,1279,557])
issue(91,'UNCERTAIN_CLASS',[531,325,572,401],
      'Nguoi cuoi phuong tien hai banh: rider ro, nhung can mentor xac nhan motorcycle hay bicycle do kich thuoc nho.')

# Frame 92: dark residential street, parked cars and curb.
box(92,'car',[576,203,600,254],True)
box(92,'car',[599,206,619,246],True)
box(92,'car',[620,208,641,242],True)
polygon(92,'direct',[533,525,607,413,662,315,689,252,710,226,747,226,
    764,235,780,240,814,252,859,273,882,286,959,339,1088,369,
    1141,439,1206,526,1128,530,953,536,756,540])
line(92,'single white',[764,286,775,300])
line(92,'road curb',[130,328,221,315,316,301,388,291,430,276])
issue(92,'UNCERTAIN_BOUNDARY',[530,220,1130,530],
      'Dem toi: ranh giua hai chieu duong va extent xe do xa khong ro; can reviewer kiem tra direct va completeness.')

# Frame 93: LOWES utility van, not a school bus, and one centre double yellow.
edit(80221,name='truck',pts=[620,321,750,464])
polygon(93,'direct',[303,642,397,568,482,503,544,451,588,407,
    611,395,621,463,749,465,750,448,776,464,811,506,856,552,934,623,
    1024,643,831,646,627,645])
polygon(93,'alternative',[753,446,789,431,811,433,852,437,881,460,
    946,463,955,532,1113,539,1206,602,1279,638,1279,670,1024,643,
    934,623,856,552,811,506,776,464])
line(93,'double yellow',[236,639,352,559,459,484,541,425,579,395])
line(93,'single white',[747,498,797,533])
line(93,'single white',[772,465,788,477])
issue(93,'UNCERTAIN_CLASS',[620,320,751,466],
      'Xe vang logo LOWES: da loai box car/bus trung, chon truck theo than cho hang; can mentor xac nhan van cho hang hay bus.')

# Frame 94: signals and intersection, observed paint only.
for p in [[554,249,575,281], [826,314,844,340]]:box(94,'traffic light',p)
for p in [[682,237,706,263],[681,267,704,287]]:box(94,'traffic sign',p)
polygon(94,'direct',[125,617,253,513,374,445,443,426,588,431,590,401,
    628,398,674,409,700,446,749,489,822,540,948,614,790,612,571,610,365,612])
line(94,'single white',[820,514,872,551,932,580])
issue(94,'LOW_VISIBILITY',[277,315,1270,435],
      'Nut giao ban dem va nguoi/vat tren dao giao thong toi; can reviewer kiem tra cac instance bi che, khong gan bong phan chieu.')

# Frame 95: passenger SUV is car; curbside parking is not alternative.
edit(87327,name='car')
polygon(95,'direct',[205,536,279,470,353,408,411,357,456,322,482,298,
    510,296,548,291,582,296,608,300,641,330,688,386,751,455,841,533,
    686,537,487,545,310,556])
line(95,'single white',[842,531,751,438,676,356,621,296])
line(95,'road curb',[0,396,70,379,142,358])
issue(95,'UNCERTAIN_BOUNDARY',[200,292,620,550],
      'Chi co dau phan cach mau vang rat mo tren duong hai chieu; reviewer xac nhan ranh trai direct, khong gan lan nguoc chieu alternative.')

# Frame 96: actual car behind the crossing, no rectangle on green reflection.
edit(87341,pts=[666,350,809,466],occ=True)
box(96,'traffic light',[591,119,622,168])
box(96,'car',[1180,370,1240,412],True)
box(96,'pedestrian',[1073,358,1094,401])
box(96,'pedestrian',[1116,352,1138,401])
polygon(96,'direct',[357,636,418,569,471,507,520,450,554,413,600,391,
    652,384,665,466,810,468,813,439,845,450,895,502,948,558,1055,646,
    853,644,608,642])
polygon(96,'alternative',[0,624,120,592,294,556,369,494,442,443,
    523,411,554,413,520,450,471,507,418,569,357,636,170,665,0,676])
for p in [[286,575,319,554],[358,510,379,495],[398,471,415,459]]:line(96,'single white',p)
for p in [[109,651,154,646],[230,646,299,646],[389,644,474,642],
          [505,642,587,641],[620,642,699,644],[727,644,818,646],
          [846,647,936,650]]:line(96,'crosswalk',p)
for p in [[566,443,595,430],[611,449,632,432],[643,450,659,434],
          [815,447,822,435],[842,450,834,435],[884,452,864,436],
          [932,452,900,436],[980,452,939,437],[1032,453,980,438],
          [1082,453,1022,438],[1134,454,1065,440]]:line(96,'crosswalk',p)
issue(96,'UNCERTAIN_SCOPE',[0,262,294,594],
      'Quay ban do an co banh xe ben trai: da loai box truck vi la xe day; mentor xac nhan ngoai scope hay food truck.')
issue(96,'LOW_VISIBILITY',[637,327,879,465],
      'Phan chieu xanh tren kinh che xe that; da xoa bus false positive, box car chi bao xe that nhin thay.')

# Frame 97: crosswalk stripes, not one bounding rectangle around the crossing.
for p in [[387,603,1108,536],[306,541,942,489],[270,497,820,465],
          [244,468,710,444],[227,448,459,430]]:line(97,'crosswalk',p)
polygon(97,'direct',[87,570,136,476,231,446,328,425,374,430,467,430,
    605,432,752,429,900,422,1094,412,1279,404,1279,680,
    1107,671,906,650,692,628,457,606,257,592])
issue(97,'UNCERTAIN_BOUNDARY',[86,424,1279,682],
      'Goc nhin cheo tai giao lo: can mentor xac nhan huong direct va extent tren mat duong, khong tao alternative neu chua ro chieu.')

# Frame 98: duplicate car removed, both signal housings kept as separate objects.
edit(87361,pts=[509,315,612,408])
polygon(98,'direct',[156,571,267,499,384,438,455,399,510,381,
    513,407,611,409,612,389,640,391,663,416,718,457,790,510,898,572,
    763,570,595,568,411,566,249,570])
polygon(98,'alternative',[614,389,660,388,666,389,702,407,816,410,
    840,411,906,431,1013,476,1144,520,1279,552,1279,603,1088,587,
    898,572,790,510,718,457,663,416,640,391])
for p in [[160,438,186,427], [210,435,234,425],[264,434,287,424],
          [319,432,342,424],[375,431,396,423],[435,430,455,423],
          [497,429,515,422],[565,431,574,421],[632,433,634,421],
          [695,435,688,422],[762,438,746,423],[827,443,803,425]]:line(98,'crosswalk',p)
line(98,'road curb',[816,410,896,431,1011,476,1143,520,1279,552])
issue(98,'UNCERTAIN_BOUNDARY',[613,387,1279,604],
      'Ranh lan ben phai khong ro sau vach sang duong; reviewer xac nhan alternative hay chi la le do xe.')

# Frame 99: remove the ego bonnet box and duplicate small vehicles.
box(99,'traffic light',[514,34,548,91])
box(99,'traffic light',[1148,0,1180,34],trunc=True)
box(99,'pedestrian',[1220,275,1245,362])
polygon(99,'direct',[161,496,299,436,447,376,574,346,639,315,
    646,315,728,315,729,298,758,297,760,316,814,316,815,303,828,315,
    828,392,882,416,965,450,1081,495,1200,528,999,522,810,507,
    591,499,381,493])
for p in [[37,509,172,500],[229,509,350,504],[434,515,544,509],
          [633,522,738,516],[838,532,940,525],[1050,543,1134,539]]:line(99,'crosswalk',p)
for p in [[597,340,622,327],[654,340,672,327],[711,342,719,329],
          [766,344,764,330],[821,346,809,332]]:line(99,'crosswalk',p)
line(99,'single white',[603,309,637,280])
issue(99,'LOW_VISIBILITY',[647,244,864,318],
      'Nhieu xe xa bi che va den loe tai giao lo; can reviewer kiem tra so instance, ranh box va occluded.')


def clean(item):
    fields = ('id', 'label_id', 'type', 'frame', 'group', 'source', 'occluded',
              'outside', 'z_order', 'rotation', 'points', 'attributes', 'elements')
    return {key: copy.deepcopy(item[key]) for key in fields if key in item}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate(shapes):
    def simple(pts, closed):
        def orient(a,b,c):
            return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        def on(a,b,p):
            return (abs(orient(a,b,p)) < 1e-7 and
                    min(a[0],b[0]) <= p[0] <= max(a[0],b[0]) and
                    min(a[1],b[1]) <= p[1] <= max(a[1],b[1]))
        def intersects(a,b,c,d):
            x,y,z,t=orient(a,b,c),orient(a,b,d),orient(c,d,a),orient(c,d,b)
            return ((x*y < 0 and z*t < 0) or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b))
        edges=list(zip(pts,pts[1:]+pts[:1] if closed else pts[1:]))
        for i,(a,b) in enumerate(edges):
            assert a != b
            for j in range(i+2,len(edges)):
                if closed and i==0 and j==len(edges)-1: continue
                if intersects(a,b,*edges[j]):return False
        return True
    fingerprints = set()
    for item in shapes:
        name = names[item['label_id']]
        assert name in allowed
        assert 80 <= item['frame'] <= 99
        p = item['points']
        assert all(0 <= value <= (1280 if i % 2 == 0 else 720) for i, value in enumerate(p))
        if name in objects:
            assert item['type'] == 'rectangle' and len(p) == 4
            assert p[0] < p[2] and p[1] < p[3]
        elif name.startswith('area/'):
            assert item['type'] == 'polygon' and len(p) >= 6
            pts=list(zip(p[::2],p[1::2]))
            area=abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(pts,pts[1:]+pts[:1]))) / 2
            assert simple(pts,True) and area > 0, (item['frame'], name, pts)
        else:
            assert item['type'] == 'polyline' and len(p) >= 4
            assert simple(list(zip(p[::2],p[1::2])),False)
        key = (item['frame'], item['label_id'], item['type'], tuple(p))
        assert key not in fingerprints
        fingerprints.add(key)
        expected = {a['id'] for a in labels[name]['attributes']}
        assert {a['spec_id'] for a in item['attributes']} == expected
    for f in range(80, 100):
        assert any(x['frame'] == f and names[x['label_id']] == 'area/direct' for x in shapes)


def render(shapes, prefix='review'):
    folder = ROOT / prefix
    folder.mkdir(exist_ok=True)
    for f in range(80, 100):
        im = Image.open(ROOT / f'frame_{f}.jpg').convert('RGBA')
        fill = Image.new('RGBA', im.size)
        d = ImageDraw.Draw(fill)
        for item in shapes:
            if item['frame'] != f: continue
            name = names[item['label_id']]
            pts = list(zip(item['points'][::2], item['points'][1::2]))
            if item['type'] == 'polygon':
                color = (30, 230, 80, 42) if name == 'area/direct' else (255, 175, 0, 42)
                d.polygon(pts, fill=color, outline=color[:3] + (230,), width=2)
        im = Image.alpha_composite(im, fill).convert('RGB')
        d = ImageDraw.Draw(im)
        for i, item in enumerate(shapes):
            if item['frame'] != f: continue
            name = names[item['label_id']]
            pts = list(zip(item['points'][::2], item['points'][1::2]))
            if item['type'] == 'rectangle':
                d.rectangle(item['points'], outline=(0, 255, 70), width=1)
                d.text(pts[0], f"{item.get('id', 'N'+str(i))}:{name}", fill='yellow', stroke_width=1, stroke_fill='black')
            elif item['type'] == 'polyline':
                d.line(pts, fill=(255, 40, 220) if name == 'lane/crosswalk' else (30, 235, 255), width=3)
        im.save(folder / f'{f}.jpg')


def main():
    final = [clean(x) for x in kept.values()] + created
    validate(final)
    deletes = [clean(old[key]) for key in old if key not in kept]
    updates = [clean(value) for key, value in kept.items() if clean(old[key]) != clean(value)]
    plan = {'job_id':206,'task_id':71,'guideline_sha256':'ac12f94c05be63d26e7f241d4e6f937c6672baafe8c7163a5e52eb09b99582df',
        'baseline_annotations_sha256':digest(baseline['annotations']),
        'delete':deletes,'update':updates,'create':created,'issues':issues,
        'deletion_reasons':reasons,'final':final,
        'counts':{'before':len(old),'delete':len(deletes),'update':len(updates),
                  'create':len(created),'final':len(final),'issues':len(issues)},
        'geometry_counts':dict(Counter(x['type'] for x in final)),
        'label_counts':dict(Counter(names[x['label_id']] for x in final))}
    (ROOT / 'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    render(final)
    print(json.dumps({k:plan[k] for k in ['counts','geometry_counts','label_counts']},ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
