"""Train the AI models shipped with Kinetra:  python packaging/pretrain.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kinetra.ai.trainer import train_all, PRETRAINED

t0 = time.time()
info = train_all(n_body_seq=250, n_windows=40000, epochs=25, n_hand=40000, K=48, out_dir=PRETRAINED,
                 progress=lambda f, t: print(f"{f:.2f} {t} {time.time() - t0:.0f}s", flush=True))
print(info, flush=True)
