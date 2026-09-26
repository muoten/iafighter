"""Replace the chorus lead in work/baute/song.json with the consensus melody of all its repetitions
(work/baute/chorus_template.json: 8th-note slots around the beat nearest 'corazón'). pyin loses the duet there."""
import json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
S = json.load(open(ROOT / 'work/baute/song.json')); T = json.load(open(ROOT / 'work/baute/chorus_template.json'))
beats = np.array(S['beats']); bd = float(np.median(np.diff(beats)))
anchors = [w['s'] for w in S['words'] if w['w'].lower().startswith('coraz')]
lead = S['lead']
for c in anchors:
    b = float(beats[np.argmin(np.abs(beats - c))]); t0, t1 = b + T['slots'][0] * bd / 2, b + (T['slots'][-1] + 1) * bd / 2
    lead = [n for n in lead if not (t0 - .01 <= n['t'] < t1)]
    new = []
    for j, p in zip(T['slots'], T['cons']):
        if p is None: continue
        s = b + j * bd / 2
        if new and new[-1]['p'] == p and abs(new[-1]['t'] + new[-1]['d'] - s) < .02 and new[-1]['d'] < bd * .9: new[-1]['d'] = round(new[-1]['d'] + bd / 2, 3)
        else: new.append({'t': round(s, 3), 'd': round(bd / 2, 3), 'p': int(p)})
    lead += new
S['lead'] = sorted(lead, key=lambda n: n['t']); S['chorus_fixed'] = anchors
json.dump(S, open(ROOT / 'work/baute/song.json', 'w'))
print('chorus replaced at', anchors, '->', len(S['lead']), 'lead notes')
