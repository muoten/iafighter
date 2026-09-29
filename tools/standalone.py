#!/usr/bin/env python3
"""tools/standalone.py -- one self-contained HTML (every img/ and audio file inlined as a data URI),
so the game opens from ~/Downloads in Chrome with no server.

    python tools/standalone.py [out.html] [--cast <pair>]
"""
import base64, mimetypes, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
out = Path(args[0] if args else Path.home() / "Downloads/ia_fighter.html").expanduser()
s = (ROOT / "index.html").read_text()


def inline(m):
    q, path = m.group(1), m.group(2)
    f = ROOT / path.split("?")[0]
    if not f.exists():
        return m.group(0)
    mt = mimetypes.guess_type(f.name)[0] or ("image/webp" if f.suffix == ".webp" else "application/octet-stream")
    return f"{q}data:{mt};base64,{base64.b64encode(f.read_bytes()).decode()}{q}"


s = re.sub(r"""(['"])((?:game/)?[\w./-]+\.(?:png|jpg|webp|mp3)(?:\?[^'"]*)?)\1""", inline, s)
if "--cast" in sys.argv:
    s = s.replace("<script>", f"<script>window.DEFAULT_CAST = '{sys.argv[sys.argv.index('--cast') + 1]}';</script>\n<script>", 1)
out.write_text(s)
print(out, f"{out.stat().st_size / 1e6:.1f} MB")
