"""Cut a head (hair + face) out of a photo with a human-parsing model, for the fighters.
usage: face_matte.py IMG x0 y0 x1 y1 OUT.png [--mirror] [--hair-below-chin 0.15] [--hair]
Long hair is cut a little below the chin so the head stays head-shaped. OUT faces RIGHT (use --mirror if the photo looks left)."""
import sys
import numpy as np, torch
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

args = sys.argv[1:]; path, box, out = args[0], tuple(map(int, args[1:5])), args[5]
mirror = '--mirror' in args
below = float(args[args.index('--hair-below-chin') + 1]) if '--hair-below-chin' in args else .15
proc = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
img = Image.open(path).convert('RGB').crop(box)
with torch.no_grad(): lg = model(**proc(images=img, return_tensors="pt")).logits
seg = torch.nn.functional.interpolate(lg, size=img.size[::-1], mode="bilinear", align_corners=False).argmax(1)[0].numpy()
face = seg == 11
keep = np.isin(seg, [2, 3, 11]); lab, n = ndimage.label(keep); keep = lab == 1 + np.argmax(ndimage.sum(keep, lab, range(1, n + 1)))
keep = ndimage.binary_fill_holes(ndimage.binary_closing(keep, np.ones((15, 15)), iterations=2))
fy = np.where(face.any(1))[0]
if len(fy):                                                     # long hair: stop a bit below the chin
    top, chin = fy.min(), fy.max(); keep[int(chin + below * (chin - top)):] = False
keep = ndimage.binary_dilation(keep, iterations=3)
lab, n = ndimage.label(keep); keep = lab == 1 + np.argmax(ndimage.sum(keep, lab, range(1, n + 1)))
a = Image.fromarray((keep * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
p = img.convert('RGBA'); p.putalpha(a); box = p.getbbox(); p = p.crop(box)
if mirror: p = ImageOps.mirror(p)
hair = Image.fromarray(((seg == 2) * 255).astype(np.uint8)).crop(box)          # --hair: the HAIR class, on the same crop/scale/placement
if mirror: hair = ImageOps.mirror(hair)
W, H = 132, 156; s = min(W / p.width, H / p.height); p = p.resize((round(p.width * s), round(p.height * s)), Image.LANCZOS)
o = Image.new('RGBA', (W, H)); o.alpha_composite(p, ((W - p.width) // 2, H - p.height)); o.save(out)
if '--hair' in args:                                            # for B/W sources (stylize_heads.GRADMAP): hair gets its own ramp
    hm = Image.new('L', (W, H)); hm.paste(hair.resize(p.size, Image.LANCZOS), ((W - p.width) // 2, H - p.height)); hm.save(out.replace('.png', '_hair.png'))
vis = np.array(img).astype(float); vis[~keep] = vis[~keep] * .35 + np.array([255, 0, 255]) * .65
Image.fromarray(vis.astype(np.uint8)).resize((img.width * 360 // img.height, 360)).save(out.replace('.png', '_vis.jpg'))
print('ok', out, img.size)
