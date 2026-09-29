"""Parody lyrics on the record's phrase timing -> game/karaoke.json.
Each new line takes the span of one original line (Whisper word times); its words are spread over the
original words' onsets, so the highlight follows the melody's rhythm."""
import json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
S = json.load(open(ROOT / 'work/baute/song.json')); W = S['words']
SETS = {  # (index of the original line's first word, its word count, new line)
  'ia': [
    (1, 7,  'Sabes que estoy colgando en la IA-A-A'),
    (8, 6,  'así que no me dejes perder'),
    (14, 7, 'Sabes que estoy colgando en la IA-A-A'),
    (21, 8, 'Te escribo poemas con IA de Afra'),
    (29, 5, 'te mando canciones y graciosidades'),
    (34, 7, 'te mando las fotos de Congui en guardia'),
    (41, 5, 'y cuando le diste la patada giratoria'),
    (46, 7, 'y así me recuerdes, IA_FRA, en tu mente'),
    (53, 8, 'que mi corazón está colgando en la IA'),
    (61, 2, '¡Cuidado, cuidado!'),
    (63, 8, 'que mi corazón está colgando en la IA-A-A'),
  ],
  'podemos': [                                                      # the Podemos parody: in their hands, the heavens, Galapagar
    (1, 7,  'Las luchas sociales... colgando en sus MA-NO-O-OS'),
    (8, 6,  'salgo del piso de Vallecas'),
    (14, 7, 'Las luchas sociales... colgando en sus MA-NO-O-OS'),
    (21, 8, 'Te prometí asaltar los cielos, compañero'),
    (29, 5, 'me corté la coleta y me fui a la tele'),
    (34, 7, 'Errejón se marchó, Yolanda nos restó'),
    (41, 5, 'y en Vistalegre, patada giratoria'),
    (46, 7, 'y en el chalet de Galapagar, sí se puede'),
    (53, 8, 'que mi corazón está colgando en sus manos'),
    (61, 2, '¡Sí se puede, sí se puede!'),
    (63, 8, 'que mi corazón está colgando en sus MA-NO-O-OS'),
  ],
}
import sys
LINES = SETS[sys.argv[1] if len(sys.argv) > 1 else 'podemos']
VOW = set('aeiouáéíóúü'); STRONG = set('aeoáéíóú')
INSEP = {'bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr', 'tr', 'ch', 'll', 'rr'}
def syllables(word):
    """Spanish syllabification, good enough for karaoke: CV splits, inseparable clusters, hiatus on two strong vowels."""
    w = word.lower(); parts, cur, i = [], '', 0
    idx = [k for k, ch in enumerate(w) if ch in VOW]
    if not idx: return [word]
    nuclei = [[idx[0]]]
    for k in idx[1:]:
        prev = nuclei[-1][-1]
        if k == prev + 1 and not (w[prev] in STRONG and w[k] in STRONG) and not (w[prev] == 'i' and w[k] == 'a' and w.startswith('ia')): nuclei[-1].append(k)
        else: nuclei.append([k])
    cuts = []
    for a, b in zip(nuclei, nuclei[1:]):
        cons = list(range(a[-1] + 1, b[0]))
        if len(cons) == 0: cuts.append(b[0])
        elif len(cons) == 1: cuts.append(cons[0])
        elif len(cons) == 2: cuts.append(cons[0] if w[cons[0]:cons[0] + 2] in INSEP else cons[1])
        else: cuts.append(cons[-2] if w[cons[-2]:cons[-2] + 2] in INSEP else cons[-1])
    pieces, last = [], 0
    for c in cuts: pieces.append(word[last:c]); last = c
    pieces.append(word[last:]); return [x for x in pieces if x]
LEAD = S['lead']
SPEED = json.load(open(ROOT / 'work/baute/arrange.json'))['speed']       # the chip plays faster, at the game's pulse
rel = lambda t: round((float(t) - S['t0']) / SPEED, 3)
out = []
spans = [(W[i0]['s'], W[i0 + n - 1]['e']) for i0, n, _ in LINES]
for li, (i0, n, text) in enumerate(LINES):
    s0 = spans[li][0] - .15; s1 = spans[li + 1][0] - .15 if li + 1 < len(LINES) else spans[li][1] + .3
    onsets = [x['t'] for x in LEAD if s0 <= x['t'] < s1] or [spans[li][0]]
    words = [(wd, syllables(wd.replace('-', ' ').replace('_', ' ').split()[0]) if False else sy) for wd, sy in
             [(wd, [p for part in wd.replace('_', '-').split('-') for p in syllables(part)]) for wd in text.split()]]
    syl = [(wi, p) for wi, (wd, parts) in enumerate(words) for p in parts]
    k, m = len(syl), len(onsets)
    if m >= k: times = [onsets[int(round(j))] for j in np.linspace(0, m - 1, k)]
    else:                                                          # more syllables than notes: split notes evenly
        ext = onsets + [s1]; times = []
        per = [k // m + (1 if j < k % m else 0) for j in range(m)]
        for j in range(m): times += list(np.linspace(ext[j], ext[j + 1], per[j] + 1)[:-1])
    ws = [[wd, []] for wd, _ in words]
    for (wi, p), t in zip(syl, times): ws[wi][1].append([p, rel(t)])
    out.append({'s': rel(times[0]), 'e': rel(min(s1, spans[li][1] + .4)), 'w': ws})
json.dump(out, open(ROOT / 'game/karaoke.json', 'w'), ensure_ascii=False, indent=1)
g = ROOT / 'index.html'; h = g.read_text()                        # and into the game itself (const KARAOKE = [...];)
i0 = h.index('const KARAOKE = ') + len('const KARAOKE = '); i1 = h.index(';', h.index(']}]', i0))
g.write_text(h[:i0] + json.dumps(out, ensure_ascii=False, separators=(',', ':')) + h[i1:])
for l in out: print(' '.join('-'.join(p for p, _ in w[1]) for w in l['w']), '|', len([1 for w in l['w'] for _ in w[1]]), 'syl')
