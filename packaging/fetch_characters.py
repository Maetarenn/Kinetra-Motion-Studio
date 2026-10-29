"""Download sample rigged characters for testing the importer:  python packaging/fetch_characters.py

Nothing is bundled with Kinetra because these models have their own licences. Read them before using a model in a
product:
  * Michelle.glb / Xbot.glb are Mixamo (Adobe) characters hosted in the three.js examples. Mixamo content is royalty-free
    to use in your own projects but may not be redistributed as-is, and Adobe's terms are the authority.
  * facecap.glb is a head with the 52 ARKit blend shapes (use it as the face character in the Face tab). Check its
    licence in the three.js repository before shipping it.
For your own product use a model you made (VRoid Studio / Blender), bought, or that is CC0 / properly licensed."""
import sys
import urllib.request
from pathlib import Path

DEST = Path.home() / ".kinetra" / "characters"
BASE = "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/"
FILES = ("Michelle.glb", "Xbot.glb", "facecap.glb")

if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DEST / name
        if target.exists():
            print("exists ", target)
            continue
        print("download", name, "...", flush=True)
        urllib.request.urlretrieve(BASE + name, target)
    print(f"\nDone. In Kinetra: Tools > Load body model ... and pick a file from {DEST}\n(facecap.glb: Tools > Load character (.glb) for the Face tab)")
    sys.exit(0)
