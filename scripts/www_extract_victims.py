"""E-B candidate extraction (PREREG deviation, recorded): the PREREG keyed candidate clusters on post_entities
org mentions, but those exist only for the earlier Canada-labelled posts. Victim organisations are therefore
extracted from every ransomware-topic P_scale root with the same local model as topic_v2s (Qwen3-4B-Instruct-2507,
greedy), from post text only, before any registry matching.

    python scripts/www_extract_victims.py --base-url http://127.0.0.1:8012/v1
        → outputs/interaction_experiment/www_eval/victims.jsonl
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path[:0] = ["src", "scripts"]
from ccint import db  # noqa: E402
from ccint.interaction import manifest as mf  # noqa: E402

PROMPT = """You read one social-media post about cybersecurity.
If the post reports that a specific organisation was hit by ransomware, extortion or a data-theft claim, return the
organisation's name exactly as written in the post. If several are named, return all. If none is named, return [].
Answer with JSON only: {"victims": ["..."]}

Post:
"""


def ask(client, url, text):
    for _ in range(3):
        try:
            r = client.post(url + "/chat/completions", json={
                "model": "user", "temperature": 0, "max_tokens": 120,
                "messages": [{"role": "user", "content": PROMPT + text[:1500]}]}, timeout=60)
            out = r.json()["choices"][0]["message"]["content"].strip()
            out = out[out.find("{"): out.rfind("}") + 1]
            v = json.loads(out).get("victims", [])
            return [x.strip() for x in v if isinstance(x, str) and x.strip()]
        except Exception:  # noqa: BLE001
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    a = ap.parse_args()
    m = mf.load("www_scale")
    ids = [int(r["post_id"]) for r in m.population("in_scale")]
    with db.connect() as c:
        rows = c.execute("""SELECT p.post_id, p.text FROM posts p JOIN post_labels t ON t.post_id = p.post_id
                            AND t.label_version = 'topic_v2s' WHERE t.topic_key = 'ransomware' AND p.post_id = ANY(%s)""",
                         (ids,)).fetchall()
    out = Path("outputs/interaction_experiment/www_eval/victims.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client() as client, ThreadPoolExecutor(24) as ex:
        res = list(ex.map(lambda r: (r["post_id"], ask(client, a.base_url, r["text"] or "")), rows))
    with out.open("w") as f:
        for pid, v in res:
            f.write(json.dumps({"post_id": pid, "victims": v}) + "\n")
    print("posts", len(res), "failed", sum(v is None for _, v in res), "with victim", sum(bool(v) for _, v in res))


if __name__ == "__main__":
    main()
