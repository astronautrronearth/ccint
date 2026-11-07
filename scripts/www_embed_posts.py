"""Embeddings for E5 (PREREG_DOSE2 amendment 4): BAAI/bge-small-en-v1.5, normalised, for every root of the
eleven settings (www_scale topic roots, www_contrast, www_contrast2, www_dose2).

    .venv/bin/python scripts/www_embed_posts.py --export        # texts -> embed/<manifest>.texts.jsonl (needs the DB)
    .venv-embed/bin/python scripts/www_embed_posts.py           # → outputs/interaction_experiment/embed/<manifest>.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np


OUT = Path("outputs/interaction_experiment/embed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="www_scale,www_contrast,www_contrast2,www_dose2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    import json
    OUT.mkdir(parents=True, exist_ok=True)
    if a.export:
        sys.path[:0] = ["src"]
        from ccint import db
        from ccint.interaction import manifest as mf
        for name in a.manifests.split(","):
            ids = sorted({int(r["post_id"]) for r in mf.load(name).rows})
            with db.connect() as c, open(OUT / f"{name}.texts.jsonl", "w") as f:
                for r in c.execute("SELECT post_id, text FROM posts WHERE post_id = ANY(%s) ORDER BY post_id", (ids,)):
                    f.write(json.dumps({"post_id": r["post_id"], "text": (r["text"] or "")[:1000]}) + "\n")
            print("exported", name, flush=True)
        return
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=a.device)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in a.manifests.split(","):
        p = OUT / f"{name}.npz"
        if p.exists():
            print("skip", name)
            continue
        rows = [json.loads(l) for l in open(OUT / f"{name}.texts.jsonl")]
        txt = {r["post_id"]: r["text"] for r in rows}
        ids = sorted(txt)
        emb = model.encode([txt[i][:1000] for i in ids], batch_size=256, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True).astype(np.float16)
        np.savez_compressed(p, ids=np.array(ids, dtype=np.int64), emb=emb)
        print(name, len(ids), emb.shape, flush=True)


if __name__ == "__main__":
    main()
