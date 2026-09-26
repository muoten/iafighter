"""'Te envío poemas…' / 'Te envío canciones de 440': pitches from a published recorder transcription
(partiturasykaraokesflautadulce.blogspot.com, DO FA MI RE… in F, moved to the record's F#), one note per 8th
from the beat nearest 'poemas' (the three repetitions measured in work/baute/poemas_template.json agree with it)."""
import json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
S = json.load(open(ROOT / 'work/baute/song.json'))
LINE1 = [61, 66, 65, 63, 61, 61, 61, 66, 61, 61, 61, 61]            # Do♯ Fa♯ Fa Re♯ Do♯ Do♯ Do♯ Fa♯ Do♯ Do♯ Do♯ Do♯
LINE2 = [61, 70, 68, 66, 70, 68, 66, 70, 68, 66, 70, 66]            # Do♯ La♯ Sol♯ Fa♯ La♯ Sol♯ Fa♯ La♯ Sol♯ Fa♯ La♯ Fa♯
beats = np.array(S['beats']); bd = float(np.median(np.diff(beats)))
anchor = [w['s'] for w in S['words'] if w['w'].lower().startswith('poema')][0]
b = float(beats[np.argmin(np.abs(beats - anchor))])
t0, t1 = b - 3 * bd / 2, b + 23 * bd / 2
new = [{'t': round(b + j * bd / 2, 3), 'd': round(bd / 2, 3), 'p': p} for j, p in zip(range(-3, 9), LINE1)]
new += [{'t': round(b + j * bd / 2, 3), 'd': round(bd / 2 * (2 if k == 11 else 1), 3), 'p': p} for k, (j, p) in enumerate(zip(range(11, 23), LINE2))]
S['lead'] = sorted([n for n in S['lead'] if not (t0 - .01 <= n['t'] < t1)] + new, key=lambda n: n['t'])
json.dump(S, open(ROOT / 'work/baute/song.json', 'w'))
print('poemas passage', round(t0, 2), '-', round(t1, 2), 'rewritten:', len(new), 'notes')
