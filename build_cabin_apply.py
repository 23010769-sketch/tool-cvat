"""Package reviewed shapes for same-origin, additive CVAT writes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'outputs' / 'cabin'
snap = json.loads((ROOT / 'live_snapshot.json').read_text(encoding='utf-8'))
plan = json.loads((ROOT / 'plan.json').read_text(encoding='utf-8'))
script = r'''
void (async (copyOutput) => {
  let stage = 'preflight';
  try {
    if (location.origin !== 'http://localhost:8080' || location.pathname !== '/tasks/11/jobs/11') throw Error('Wrong CVAT job');
    if (window.__cabinWriting) throw Error('A write is already in progress');
    const plan = __PLAN__;
    const expected = __SNAPSHOT__;
    const read = async path => {
      const r = await fetch(path, {credentials:'same-origin'});
      if (!r.ok) throw Error(path + ': HTTP ' + r.status);
      return r.json();
    };
    const stable = v => JSON.stringify(v);
    const labelKey = l => [l.id,l.name,l.type,l.parent_id ?? null,
      (l.attributes||[]).map(a=>[a.id,a.name,a.input_type,a.default_value,a.values]).sort((a,b)=>a[0]-b[0]),
      (l.sublabels||[]).map(labelKey).sort((a,b)=>a[0]-b[0])];
    const attrs = a => (a||[]).map(v=>[v.spec_id,v.value]).sort((a,b)=>a[0]-b[0]);
    const tagKey = s => [s.frame,s.label_id,s.group||0,attrs(s.attributes)];
    const shapeKey = s => [s.type,s.frame,s.label_id,s.group||0,s.points||[],
      !!s.occluded,!!s.outside,s.rotation||0,s.z_order||0,attrs(s.attributes),
      (s.elements||[]).map(shapeKey).sort((a,b)=>a[2]-b[2])];
    const sameList = (a,b,key) => stable(a.map(key).map(stable).sort()) === stable(b.map(key).map(stable).sort());
    const [job,labels,meta] = await Promise.all([
      read('/api/jobs/11'), read('/api/labels?job_id=11&page_size=100'), read('/api/jobs/11/data/meta')]);
    if (job.id !== 11 || job.task_id !== 11 || job.task_name !== 'Day12-G07' || job.frame_count !== 22) throw Error('Job identity changed');
    if (!sameList(labels.results, expected.labels.results, labelKey)) throw Error('Label schema changed');
    if (stable(meta.frames) !== stable(expected.meta.frames) || stable(meta.deleted_frames)!==stable(expected.meta.deleted_frames)) throw Error('Frame metadata changed');
    let current = await read('/api/jobs/11/annotations');
    if (current.tracks.length || (current.intervals||[]).length) throw Error('Unexpected existing tracks/intervals');
    if (current.tags.length && !sameList(current.tags,plan.tags,tagKey)) throw Error('Existing tags differ; no data overwritten');
    if (current.shapes.length && !sameList(current.shapes,plan.shapes,shapeKey)) throw Error('Existing shapes differ; no data overwritten');
    const csrf = document.cookie.split('; ').find(v=>v.startsWith('csrftoken='))?.slice('csrftoken='.length);
    if (!csrf) throw Error('CSRF protection unavailable');
    const create = async body => {
      const r = await fetch('/api/jobs/11/annotations?action=create', {
        method:'PATCH', credentials:'same-origin',
        headers:{'Content-Type':'application/json','X-CSRFToken':decodeURIComponent(csrf)},
        body: JSON.stringify(body)});
      if (!r.ok) throw Error('Create returned HTTP '+r.status+': '+(await r.text()).slice(0,1500));
      return r.json();
    };
    window.__cabinWriting = true;
    if (!current.tags.length) {
      stage='creating_frame_tags';
      await create({version:current.version,tags:plan.tags,shapes:[],tracks:[]});
      current=await read('/api/jobs/11/annotations');
      if (!sameList(current.tags,plan.tags,tagKey)) throw Error('Tag verification failed; stopped before shapes');
    }
    if (!current.shapes.length) {
      stage='creating_reviewed_shapes';
      await create({version:current.version,tags:[],shapes:plan.shapes,tracks:[]});
    }
    stage='verifying';
    current=await read('/api/jobs/11/annotations');
    if (!sameList(current.tags,plan.tags,tagKey) || !sameList(current.shapes,plan.shapes,shapeKey) || current.tracks.length) throw Error('Full readback mismatch');
    copyOutput(JSON.stringify({ok:true,stage:'verified',job_id:11,task_name:job.task_name,annotations:current}));
  } catch (e) {
    copyOutput(JSON.stringify({ok:false,stage,error:String(e)}));
  } finally {
    window.__cabinWriting=false;
  }
})(copy);
'''
# Only scope metadata and labels are embedded, never image content or auth.
expected = {k:snap[k] for k in ('labels','meta')}
script = script.replace('__PLAN__',json.dumps(plan,separators=(',',':'))).replace('__SNAPSHOT__',json.dumps(expected,separators=(',',':')))
(ROOT / 'apply.js').write_text(script,encoding='utf-8')
print('Prepared additive write:',len(plan['tags']),'tags,',len(plan['shapes']),'shapes')
