"""Recolour Ken's SSF2 sprite sheet (work/ref/ken.png) into Congui and IA_FRA atlases.

Palette swap the way SF2 does alternate costumes: every gi tone maps, by rank, onto the
fighter's own 6-tone ramp (top vs trousers split at the black belt), skin onto their skin
ramp; Ken's hair is removed and each frame's head centre is saved for the photo face.
Out: game/img/sprites_<who>.png + game/sprites.json"""
import json, pathlib
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = pathlib.Path(__file__).resolve().parent.parent
ken = Image.open(ROOT / 'work/ref/ken.png').convert('RGBA'); K = np.array(ken).astype(int)
ROWS = {'walk': (0, 4), 'idle': (1, 4), 'punch': (2, 3), 'kick': (6, 5), 'spin': (7, 5), 'jump': (8, 7), 'crouch': (9, 1)}
FW, FH = 70, 80

r, g, b, al = K[..., 0], K[..., 1], K[..., 2], K[..., 3] > 0
gi = al & (r >= 55) & (g <= 80) & (b <= 12) & (r > g * 1.6)
hair = al & (((b < 20) & (g > 140) & (r > 200)) | ((g > 215) & (r > 230)))
skin = al & ~gi & ~hair & (r > g) & (g >= b) & (r >= 60) & (r - b > 40)
dark = al & (r < 40) & (g < 40) & (b < 40)

def ramp_index(px, ramp_src):                        # nearest source tone → rank in the ramp
    d = ((px[:, None, :] - ramp_src[None]) ** 2).sum(-1); return d.argmin(1)
GI = np.array([[64, 0, 0], [96, 16, 0], [136, 32, 0], [176, 48, 0], [216, 64, 0], [249, 72, 0]])
SK = np.array([[72, 40, 24], [128, 64, 48], [184, 104, 72], [204, 153, 102], [241, 192, 128]])
hexs = lambda *h: np.array([[int(x[i:i + 2], 16) for i in (1, 3, 5)] for x in h])
LOOK = {
  'congui': dict(top=hexs('#3c3c48', '#6e6e7c', '#9d9daa', '#c6c6cf', '#e6e6ec', '#ffffff'),
                 pants=hexs('#1d2e49', '#2e4a73', '#46699c', '#648bbd', '#88abd6', '#abc8ea'),
                 skin=hexs('#58301f', '#8c503b', '#c47e5e', '#dca680', '#f4cca0'), belt=(38, 32, 30)),
  'iafra':  dict(top=hexs('#08070b', '#131118', '#1f1c26', '#2d2936', '#403b4c', '#58526a'),
                 pants=hexs('#0e183a', '#18295a', '#243d7d', '#3353a0', '#476cbe', '#6689d4'),
                 skin=hexs('#643826', '#9a5c46', '#cf8e70', '#e8b896', '#fad8ba'), belt=(92, 62, 36)),
}

# hand-set jacket/trouser splits where the body is tipped or seen from behind (read off the enlarged frames)
SPLIT = {('spin', 0): lambda x, y: y < 46, ('spin', 1): lambda x, y: x < 28,
         ('spin', 2): lambda x, y: x < 26, ('spin', 3): lambda x, y: (x < 31) & (y < 45)}
# head centres per frame (sprite px), read off the enlarged frames: the hair box alone drifts onto arms
HEAD = {'walk': [(29, 12), (36, 13), (50, 15), (52, 16)], 'idle': [(38, 9), (39, 8), (39, 9), (38, 8)],
        'punch': [(37, 8), (36, 9), (38, 8)], 'kick': [(41, 8), (31, 8), (14, 9), (32, 8), (41, 8)],
        'spin': [(13, 26), (6, 27), (7, 22), (14, 20), (22, 17)],
        'jump': [(40, 12), (29, 10), (36, 8), (39, 8), (36, 8), (29, 10), (42, 12)], 'crouch': [(40, 34)]}
def paint_neck(o, head, ramp, slim=False):
    """Ken tucks his chin behind his fist: no neck to keep. Paint one in his pixel style — 1px dark outline,
    shade on the nape side, light on the throat, the jaw's shadow on top — only where the frame is empty,
    so the body stays in front. It runs from under the (raised) head down into the collar."""
    cx, top, bot = int(round(head[0] - 1)), int(round(head[1] - (4 if slim else 6))), int(round(head[1] + 10))   # top reaches under the raised head
    out = (ramp[0] * 0.55).astype(np.uint8)
    cols = [out, ramp[1], ramp[1], ramp[1], ramp[1], ramp[2], ramp[3], ramp[3], ramp[3], ramp[2], out]   # wide nape → throat
    if slim: cols = cols[1:]                                                       # a narrow profile head (Echenique): a slimmer nape, still reaching the shoulder
    for y in range(top, bot + 1):
        for k, c in enumerate(cols):
            x = cx - 6 + k + (1 if slim else 0)
            if 0 <= y < FH and 0 <= x < FW and o[y, x, 3] == 0:
                col = ramp[1] if (y - top < 5 and 0 < k < 10) else c                         # under the jaw: in shadow
                o[y, x, :3] = col; o[y, x, 3] = 255
    # trapezius: bridge any gap between the neck and the shoulders (no South Park gap), row by row
    filled = np.zeros((FH, FW), bool)
    for y in range(top + 2, min(FH, bot + 5)):
        for d, x0, tone in ((-1, cx - 7, ramp[1]), (1, cx + 5, ramp[2])):
            run = []
            for step in range(14):
                x = x0 + d * step
                if not (0 <= x < FW): run = []; break
                if o[y, x, 3] > 0: break
                run.append(x)
            else: run = []                                                        # no shoulder within reach: leave the air alone
            for x in run: o[y, x, :3] = tone; o[y, x, 3] = 255; filled[y, x] = True
    ys, xs = np.where(filled)                                                     # outline the top of the new slope
    for y, x in zip(ys, xs):
        if y == 0 or o[y - 1, x, 3] == 0: o[y, x, :3] = (ramp[0] * 0.55).astype(np.uint8)
frames = {}
out = {w: np.zeros((len(ROWS) * FH, 7 * FW, 4), np.uint8) for w in LOOK}
for ri, (name, (row, n)) in enumerate(ROWS.items()):
    frames[name] = []
    for i in range(n):
        ys, xs = slice(row * FH, row * FH + FH), slice(i * FW, i * FW + FW)
        c_al, c_gi, c_hair, c_skin, c_dark = al[ys, xs], gi[ys, xs], hair[ys, xs], skin[ys, xs], dark[ys, xs]
        # hair = only the biggest yellow blob (the head); pale yellow elsewhere is skin highlight on the arms
        hx0, hy0 = HEAD[name][i]; gy0, gx0 = np.mgrid[0:FH, 0:FW]
        near = (gx0 - hx0) ** 2 + (gy0 - hy0) ** 2 <= 13 ** 2                # hair only around the hand-set head
        stray = c_hair & ~near; c_hair = c_hair & near; c_skin = c_skin | stray
        px = K[ys, xs, :3]
        # head: the hair blob nearest the top; centre a little below/in front of it
        hl, hn = ndimage.label(ndimage.binary_dilation(c_hair, iterations=1))
        head = None
        if hn:
            sizes = ndimage.sum(c_hair, hl, range(1, hn + 1)); L = 1 + int(np.argmax(sizes))
            hy, hx = np.where((hl == L) & c_hair); head = [float(hx.mean()), float(hy.mean() + 4)]
        # belt = the near-black blob that touches the most gi; split top/trousers along its own axis (it tilts in kicks)
        dl, dn = ndimage.label(c_dark); gi_near = ndimage.binary_dilation(c_gi, iterations=1)
        best = max(range(1, dn + 1), key=lambda L: ((dl == L) & gi_near).sum(), default=0)
        by, bx = np.where(dl == best) if best else (np.array([42]), np.array([35]))
        cy_, cx_ = by.mean(), bx.mean()
        if len(bx) > 3:                                    # belt axis = its longest span (end to end), so the hanging knot tails don't tip it
            P = np.stack([bx, by], 1).astype(float); D = ((P[:, None] - P[None]) ** 2).sum(-1); i0, i1 = np.unravel_index(D.argmax(), D.shape)
            axis = (P[i1] - P[i0]) / (np.sqrt(D[i0, i1]) or 1)
        else: axis = np.array([1.0, 0.0])
        nrm = np.array([-axis[1], axis[0]])
        hd = np.array(head) if head else np.array([cx_, cy_ - 30])
        side = np.sign((hd[0] - cx_) * nrm[0] + (hd[1] - cy_) * nrm[1]) or -1
        belt_mask = dl == best if best else np.zeros_like(c_gi)
        cut = c_gi & ~ndimage.binary_dilation(belt_mask, iterations=2)     # the belt separates jacket from trousers
        lab, nl = ndimage.label(cut); top = np.zeros_like(c_gi)
        for L in range(1, nl + 1):
            yy, xx = np.where(lab == L)
            if yy.max() > FH - 14: continue                    # reaches the floor: a trouser leg, whatever the belt says
            if name != 'spin' and yy.mean() > cy_ + 2: continue   # upright poses: anything centred below the belt is trousers
            if np.sign((xx.mean() - cx_) * nrm[0] + (yy.mean() - cy_) * nrm[1]) == side: top |= lab == L
        top |= c_gi & ~cut & ndimage.binary_dilation(top, iterations=2) & ~ndimage.binary_dilation(c_gi & ~top & cut, iterations=1)   # the cut rim follows its neighbour
        if (name, i) in SPLIT: gy, gx = np.mgrid[0:FH, 0:FW]; top = c_gi & SPLIT[(name, i)](gx, gy)
        pants = c_gi & ~top
        head = list(HEAD[name][i])
        gy, gx = np.mgrid[0:FH, 0:FW]
        in_head = ((gx - head[0]) / 7.5) ** 2 + ((gy - head[1] + 1) / 8.5) ** 2 <= 1   # Ken's whole head goes, not just the hair
        tipped = name == 'spin' or (name == 'kick' and 1 <= i <= 3)

        frames[name].append({'x': i * FW, 'y': ri * FH, 'head': head, 'waist': [float(cx_), float(cy_)]})
        for who, lk in LOOK.items():
            o = np.zeros((FH, FW, 4), np.uint8); o[..., :3] = px; o[..., 3] = np.where(c_al, 255, 0)
            for mask, src, dst in [(top, GI, lk['top']), (pants, GI, lk['pants']), (c_skin, SK, lk['skin'])]:
                if mask.any(): o[mask, :3] = dst[ramp_index(px[mask], src)]
            o[belt_mask, :3] = lk['belt']
            gone = ndimage.binary_dilation(c_hair, iterations=2) & ((px.sum(-1) < 270) | ~c_gi & ~c_skin) | c_hair   # hair + its dark outline ring
            o[gone | (in_head & ~c_gi), 3] = 0                 # Ken's head goes; the photo face replaces it
            lab_, n_ = ndimage.label(o[..., 3] > 0)              # crumbs of hair/outline left around the head
            if n_ > 1:
                sz = ndimage.sum(np.ones_like(lab_), lab_, range(1, n_ + 1))
                for L in range(1, n_ + 1):
                    if sz[L - 1] < 8: o[lab_ == L, 3] = 0
            if not tipped: paint_neck(o, head, lk['skin'], slim=(who == 'iafra'))     # pixel-art neck, BEHIND the body, upscaled with it
            out[who][ri * FH:(ri + 1) * FH, i * FW:(i + 1) * FW] = o
for who, arr in out.items(): Image.fromarray(arr).save(ROOT / f'work/sprites_{who}_1x.png')

# Ken's guard leans forward (the "hunchback"): in the stance frames, tip the upper body back about the waist,
# so the chest comes up and the guard sits higher. Cut at the belt, rotate the top half, keep the legs.
UPRIGHT = {'idle': [0, 1, 2, 3], 'walk': [0, 1, 2, 3], 'punch': [0, 1, 2], 'kick': [0, 4], 'jump': [0, 6]}
TILT = int(__import__("os").environ.get("TILT", 13))                                                 # degrees back
def straighten(hd, update_json):
    S = 4; out = hd.copy()
    for name, idxs in UPRIGHT.items():
        for i in idxs:
            fr = frames[name][i]; x0, y0 = fr['x'] * S, fr['y'] * S
            cell = hd[y0:y0 + FH * S, x0:x0 + FW * S]
            px, py = fr['waist'][0] * S, fr['waist'][1] * S
            gy, gx = np.mgrid[0:FH * S, 0:FW * S]
            top = cell.copy(); top[gy > py + 6] = 0              # the upper body (a little past the belt, so the seam overlaps)
            low = cell.copy(); low[gy <= py - 2] = 0
            rot = np.array(Image.fromarray(top).rotate(TILT, resample=Image.BICUBIC, center=(px, py)))
            comp = Image.fromarray(low); comp.alpha_composite(Image.fromarray(rot))
            out[y0:y0 + FH * S, x0:x0 + FW * S] = np.array(comp)
            if update_json and not fr.get('tilted'):     # move the head with the chest
                t = np.radians(TILT); hx, hy = fr['head']; cx, cy = fr['waist']
                fr['head'] = [cx + (hx - cx) * np.cos(t) + (hy - cy) * np.sin(t), cy - (hx - cx) * np.sin(t) + (hy - cy) * np.cos(t)]
                fr['tilted'] = TILT
    return out

# HD: Real-ESRGAN (anime model) x4 on colour and on alpha separately — colour bled into the transparent area first,
# so edges don't pick up a dark fringe
import subprocess
ESR = ROOT / 'work/esr'
for who, arr in out.items():
    a = arr[..., 3] > 0
    idx = ndimage.distance_transform_edt(~a, return_distances=False, return_indices=True)
    rgb = arr[..., :3][idx[0], idx[1]]
    Image.fromarray(rgb).save(ROOT / f'work/_{who}_rgb.png'); Image.fromarray(np.repeat(arr[..., 3:], 3, 2)).save(ROOT / f'work/_{who}_a.png')
    for part in ('rgb', 'a'):
        subprocess.run([str(ESR / 'realesrgan-ncnn-vulkan'), '-i', str(ROOT / f'work/_{who}_{part}.png'), '-o', str(ROOT / f'work/_{who}_{part}_x4.png'),
                        '-n', 'realesrgan-x4plus-anime', '-s', '4', '-m', str(ESR / 'models')], check=True, capture_output=True)
    rgb4 = np.array(Image.open(ROOT / f'work/_{who}_rgb_x4.png').convert('RGB'))
    a4 = np.array(Image.open(ROOT / f'work/_{who}_a_x4.png').convert('L')).astype(float)
    a4 = np.clip((a4 - 70) / (185 - 70), 0, 1) * 255                      # firm up the upscaled edge
    hd = straighten(np.dstack([rgb4, a4.astype(np.uint8)]), who == list(LOOK)[-1])
    Image.fromarray(hd).save(ROOT / f'game/img/sprites_{who}.webp', quality=90, method=6)
json.dump({'fw': FW, 'fh': FH, 'hd': 4, 'base': {'walk': 75}, 'frames': frames}, open(ROOT / 'game/sprites.json', 'w'), indent=1)
print({k: len(v) for k, v in frames.items()})
