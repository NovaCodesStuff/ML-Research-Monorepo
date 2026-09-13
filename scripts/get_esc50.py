"""Download ESC-50 (≈600 MB) into data/raw/esc50."""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
dst = Path(__file__).resolve().parents[1] / "data" / "raw" / "esc50"
if (dst / "audio").exists():
    print("already present:", dst); sys.exit(0)
dst.parent.mkdir(parents=True, exist_ok=True)
print("downloading", URL)
buf = io.BytesIO(urllib.request.urlopen(URL).read())
with zipfile.ZipFile(buf) as z:
    z.extractall(dst.parent)
(dst.parent / "ESC-50-master").rename(dst)
print("done:", dst)
