"""E6 (PREREG_DOSE2 amendment 4): content type of every root of the eleven settings, local Qwen3-4B-Instruct-2507
(greedy) via vLLM. Labels: question, opinion, news, automated, promotion, other.

    python scripts/www_post_types.py --base-url http://127.0.0.1:8016/v1 → outputs/interaction_experiment/ptype/<manifest>.jsonl
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path[:0] = ["src"]
from ccint import db  # noqa: E402
from ccint.interaction import manifest as mf  # noqa: E402

LABELS = ["question", "opinion", "news", "automated", "promotion", "other"]
PROMPT = """Classify this social-media post by what kind of post it is. Answer with exactly one word from:
question (asks others something), opinion (personal view, comment or conversation), news (shares a news item,
headline or article link), automated (templated or bot-like feed post, e.g. price ticker, "New post:", listing
alerts), promotion (advertising, self-promotion, giveaways), other.

Post:
"""
OUT = Path("outputs/interaction_experiment/ptype")


def ask(client, url, text):
    for _ in range(3):
        try:
            r = client.post(url + "/chat/completions", json={"model": "user", "temperature": 0, "max_tokens": 4,
                            "messages": [{"role": "user", "content": PROMPT + text[:1000]}]}, timeout=60)
            w = r.json()["choices"][0]["message"]["content"].strip().lower()
            for lab in LABELS:
                if w.startswith(lab):
                    return lab
            return "other"
        except Exception:  # noqa: BLE001
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--manifests", default="www_contrast,www_contrast2,www_scale,www_dose2")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in a.manifests.split(","):
        p = OUT / f"{name}.jsonl"
        if p.exists():
            print("skip", name)
            continue
        m = mf.load(name)
        ids = sorted({int(r["post_id"]) for r in m.rows})
        with db.connect() as c:
            rows = c.execute("SELECT post_id, text FROM posts WHERE post_id = ANY(%s) ORDER BY post_id", (ids,)).fetchall()
        with httpx.Client() as client, ThreadPoolExecutor(32) as ex:
            res = list(ex.map(lambda r: (r["post_id"], ask(client, a.base_url, r["text"] or "")), rows))
        with p.open("w") as f:
            for pid, lab in res:
                f.write(json.dumps({"post_id": pid, "type": lab}) + "\n")
        print(name, len(res), "failed", sum(l is None for _, l in res), flush=True)


if __name__ == "__main__":
    main()
