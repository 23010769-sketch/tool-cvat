"""Prepare the requested narrow corrections against the latest CVAT snapshot."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'outputs' / 'cabin'
snapshot = json.loads((ROOT / 'before_corrections.json').read_text(encoding='utf-8'))
before = snapshot['annotations']
after = copy.deepcopy(before)
labels = {x['name']: x for x in snapshot['labels']['results']}
occupied_id = next(a['id'] for a in labels['child_seat']['attributes'] if a['name']=='occupied')
eye_sides = {0: ('left','right'), 1: ('left','right'), 3: ('left',), 5: ('right',)}
eye_names = {x['id']: x['name'] for x in labels['face13']['sublabels']}
updates = []
changes = []
for shape in after['shapes']:
    changed = False
    if shape['label_id'] == labels['child_seat']['id'] and shape['frame'] in (6,7,20,21):
        attr = next(a for a in shape['attributes'] if a['spec_id']==occupied_id)
        if attr['value']=='yes':
            attr['value']='unknown'
            changes.append({'id':shape['id'],'frame':shape['frame'],'change':'occupied: yes -> unknown'})
            changed=True
    if shape['label_id'] == labels['face13']['id'] and shape['frame'] in eye_sides:
        point_count=0
        for point in shape['elements']:
            name=eye_names[point['label_id']]
            if any(name.startswith(side+'_eye_') for side in eye_sides[shape['frame']]):
                point['outside']=True
                point['occluded']=True
                point_count+=1
        assert point_count == 4 * len(eye_sides[shape['frame']])
        changes.append({'id':shape['id'],'frame':shape['frame'],'change':f'{point_count} eye points: outside=true, occluded=true'})
        changed=True
    if changed:
        updates.append(shape)
assert len(updates)==8
(ROOT/'correction_plan.json').write_text(json.dumps({'changes':changes,'updates':updates},ensure_ascii=False,indent=2),encoding='utf-8')
script=r'''
void (async (copyOutput) => {
  let stage='preflight';
  try {
    if(location.origin!=='http://localhost:8080') throw Error('Wrong CVAT origin');
    const before=__BEFORE__;
    const expected=__AFTER__;
    const updates=__UPDATES__;
    const normalize=v=>{
      if(Array.isArray(v)) {
        let a=v.map(normalize);
        if(a.length && a.every(x=>x && typeof x==='object' && !Array.isArray(x))) {
          if(a.every(x=>'id' in x)) a.sort((x,y)=>x.id-y.id);
          else if(a.every(x=>'spec_id' in x)) a.sort((x,y)=>x.spec_id-y.spec_id);
        }
        return a;
      }
      if(v && typeof v==='object') return Object.fromEntries(Object.keys(v).sort().filter(k=>k!=='version').map(k=>[k,normalize(v[k])]));
      return v;
    };
    const same=(a,b)=>JSON.stringify(normalize(a))===JSON.stringify(normalize(b));
    const read=async path=>{const r=await fetch(path);if(!r.ok)throw Error(path+': '+r.status);return r.json();};
    const job=await read('/api/jobs/11');
    if(job.id!==11||job.task_id!==11||job.task_name!=='Day12-G07')throw Error('Wrong job');
    let current=await read('/api/jobs/11/annotations');
    if(!same(current,expected)) {
      if(!same(current,before))throw Error('Annotations changed after review; stopped without writing');
      const csrf=document.cookie.split('; ').find(v=>v.startsWith('csrftoken='))?.slice(10);
      if(!csrf)throw Error('CSRF unavailable');
      stage='updating_eight_existing_shapes';
      const response=await fetch('/api/jobs/11/annotations?action=update',{
        method:'PATCH',credentials:'same-origin',headers:{'Content-Type':'application/json','X-CSRFToken':decodeURIComponent(csrf)},
        body:JSON.stringify({version:current.version,tags:[],shapes:updates,tracks:[]})});
      if(!response.ok)throw Error('HTTP '+response.status+': '+(await response.text()).slice(0,1000));
      await response.json();
    }
    stage='readback_verification';
    current=await read('/api/jobs/11/annotations');
    if(!same(current,expected))throw Error('Readback differs from expected correction');
    copyOutput(JSON.stringify({ok:true,stage:'verified',annotations:current}));
  } catch(error) {copyOutput(JSON.stringify({ok:false,stage,error:String(error)}));}
})(copy);
'''
for placeholder, value in [('__BEFORE__',before),('__AFTER__',after),('__UPDATES__',updates)]:
    script=script.replace(placeholder,json.dumps(value,separators=(',',':')))
(ROOT/'correct_review.js').write_text(script,encoding='utf-8')
print(json.dumps(changes,ensure_ascii=False,indent=2))
