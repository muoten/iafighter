"""Turn the photo heads into sprite-style heads: the same treatment as the bodies.

Photo -> sprite resolution -> few-colour palette -> dark outline on the silhouette -> Real-ESRGAN anime x4.
usage: stylize_heads.py [scale]   (scale = head pixels per sprite pixel; 1 = SF2 density)"""
import subprocess, sys, pathlib
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = pathlib.Path(__file__).resolve().parent.parent
ESR = ROOT / 'work/esr'
SRC = {'congui': ROOT / 'work/face_congui_prof.png',             # profile (video frame 24): faces the opponent
       'afra': ROOT / 'work/face_afra_prof.png',
       'iglesias': ROOT / 'work/face_iglesias_prof.png',             # Wikimedia Commons, Olaf Kosinsky, CC BY-SA 3.0 de
       'echenique': ROOT / 'work/face_echenique_prof.png'}           # Wikimedia Commons, Pablo Ibáñez, CC BY-SA 2.0
INK = {'iglesias': dict(clahe=1.5, thr=6, dark=0.5), 'echenique': dict(clahe=1.3, thr=4, dark=0.35), 'congui': dict(clahe=1.8, thr=6, dark=0.45), 'afra': dict(clahe=2.5, thr=4, dark=0.25)}   # softer ink on Congui: creases read as age
NO_OVAL = {'echenique'}                                             # a wide profile: the oval would cut his chin and beard
SKIN = {'iglesias': (220, 166, 128), 'echenique': None, 'congui': (220, 166, 128), 'afra': (232, 184, 150)}          # the body's mid skin tone (make_sprites ramps)
HEAD_W, HEAD_H = 17, 20                                    # the head box in sprite pixels (as drawn in the game)

def cartoon(src, skin=None, clahe=2.5, thr=4, dark=0.25):
    """Photo -> cel look: edge-preserving colour smoothing + ink lines on the features (eyes, brows, mouth, jaw)."""
    import cv2
    im = np.array(Image.open(src).convert('RGBA')); rgb, a = im[..., :3].copy(), im[..., 3]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)                 # dim video frames: local contrast on lightness first
    lab[..., 0] = cv2.createCLAHE(clipLimit=clahe, tileGridSize=(4, 4)).apply(lab[..., 0])
    if skin is not None:                                       # pull the face's colour toward the body's skin tone
        t = cv2.cvtColor(np.uint8([[skin]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float); m = a > 128
        for ch in (1, 2): lab[..., ch] = np.clip(lab[..., ch] + 0.6 * (t[ch] - lab[..., ch][m].mean()), 0, 255)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    big = cv2.resize(rgb, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    for _ in range(4): big = cv2.bilateralFilter(big, 9, 40, 7)
    grey = cv2.cvtColor(big, cv2.COLOR_RGB2GRAY)
    g1, g2 = cv2.GaussianBlur(grey, (0, 0), 1.6).astype(float), cv2.GaussianBlur(grey, (0, 0), 1.6 * 1.6).astype(float)
    dog = g1 - 0.98 * g2                                       # XDoG: dark lines where the face has strong local contrast
    ink = np.clip((dog + thr) / 6, 0, 1)                         # 0 = line, 1 = no line
    ink = cv2.erode((ink * 255).astype(np.uint8), np.ones((2, 2), np.uint8)).astype(float) / 255
    toon = (big.astype(float) * (dark + (1 - dark) * ink[..., None])).clip(0, 255).astype(np.uint8)
    toon = cv2.resize(toon, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
    return Image.fromarray(np.dstack([toon, a]))

def stylize(src, scale, who, colors=16):
    im = cartoon(src, SKIN[who], **INK[who]); w, h = round(HEAD_W * scale), round(HEAD_H * scale)
    small = np.array(im.resize((w, h), Image.LANCZOS)).astype(int)
    a = small[..., 3] > 110
    rgb = Image.fromarray(small[..., :3].astype(np.uint8)).quantize(colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert('RGB')
    rgb = np.array(rgb).astype(float)
    yy, xx = np.mgrid[0:h, 0:w]                                # round off the flat neck cut at the bottom
    if who not in NO_OVAL: a &= ((xx - (w - 1) / 2) / (w / 2)) ** 2 + ((yy - (h - 1) * 0.42) / (h * 0.6)) ** 2 <= 1
    edge = a & ~ndimage.binary_erosion(a)                  # sprite outline: the silhouette's rim, darkened
    edge &= yy < h * 0.74                                    # ...but not under the jaw: that edge joins the neck
    rgb[edge] *= 0.45
    out = np.dstack([rgb.clip(0, 255).astype(np.uint8), np.where(a, 255, 0).astype(np.uint8)])
    return out

def upscale(arr, tag):
    a = arr[..., 3] > 0
    idx = ndimage.distance_transform_edt(~a, return_distances=False, return_indices=True)
    Image.fromarray(arr[..., :3][idx[0], idx[1]]).save(ROOT / f'work/_{tag}_rgb.png')
    Image.fromarray(np.repeat(arr[..., 3:], 3, 2)).save(ROOT / f'work/_{tag}_a.png')
    for part in ('rgb', 'a'):
        subprocess.run([str(ESR / 'realesrgan-ncnn-vulkan'), '-i', str(ROOT / f'work/_{tag}_{part}.png'), '-o', str(ROOT / f'work/_{tag}_{part}_x4.png'),
                        '-n', 'realesrgan-x4plus-anime', '-s', '4', '-m', str(ESR / 'models')], check=True, capture_output=True)
    rgb4 = np.array(Image.open(ROOT / f'work/_{tag}_rgb_x4.png').convert('RGB'))
    a4 = np.array(Image.open(ROOT / f'work/_{tag}_a_x4.png').convert('L')).astype(float)
    al = np.clip((a4 - 70) / 115, 0, 1)
    H, Wd = al.shape; yy, xx = np.mgrid[0:H, 0:Wd]                             # fade into the neck ONLY at the nape (back-bottom corner):
    ramp = np.clip((H * 0.98 - yy) / (H * 0.08), 0, 1)                         # the face — mouth, chin — stays fully opaque
    ramp = np.where(xx < Wd * 0.45, ramp, 1.0)
    return Image.fromarray(np.dstack([rgb4, (al * ramp * 255).astype(np.uint8)]))

def lite(who, colors=24):
    """Detailed faces (beard, glasses): cel-shade at full size and posterize, no sprite-resolution round trip."""
    im = cartoon(SRC[who], SKIN.get(who), **INK[who]); arr = np.array(im).astype(int); a = arr[..., 3] > 110
    rgb = np.array(Image.fromarray(arr[..., :3].astype(np.uint8)).quantize(colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert('RGB')).astype(float)
    edge = a & ~ndimage.binary_erosion(a, iterations=2); h = a.shape[0]; yy = np.arange(h)[:, None]
    rgb[edge & (yy < h * .74)] *= .45
    out = np.dstack([rgb.clip(0, 255).astype(np.uint8), np.where(a, 255, 0).astype(np.uint8)])
    return upscale(out, f'{who}_lite')
LITE = {'iglesias', 'echenique'}

if __name__ == '__main__':
    scales = [float(sys.argv[1])] if len(sys.argv) > 1 else [1.5]
    for who, src in SRC.items():
        if len(sys.argv) > 2 and who not in sys.argv[2:]: continue
        if who in LITE: lite(who).save(ROOT / f'work/head_{who}_lite.png'); continue
        for sc in scales:
            upscale(stylize(src, sc, who), f'{who}_{sc}').save(ROOT / f'work/head_{who}_{sc}.png')
