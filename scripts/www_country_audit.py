"""LLM audit of the country filters (PREREG frozen_scale): 25 random matched posts per country, labelled
by claude-sonnet-5-5 (reference) and claude-haiku-4-5 (second rater) through the relay, temperature 0.
These are LLM labels, reported as such, never as human validation.

    python scripts/www_country_audit.py   → outputs/www_audit/country/{items.csv,labels.csv,summary.json}
Relay credentials are read from the environment (source ~/.secrets/relay.env); nothing is printed.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path[:0] = ["scripts", "src"]
from ccint import db  # noqa: E402
from country_filters import COUNTRIES, countries  # noqa: E402

SEED, PER = 20261106, 25
MODELS = {"reference": "claude-sonnet-5-5", "second": "claude-haiku-4-5-20251001"}
OUT = Path("outputs/www_audit/country")
NICE = {"canada": "Canada", "united_states": "the United States", "united_kingdom": "the United Kingdom",
        "australia": "Australia", "india": "India", "germany": "Germany", "france": "France", "japan": "Japan",
        "netherlands": "the Netherlands", "singapore": "Singapore"}
PROMPT = """You are auditing a keyword filter. Read the social-media post below and decide whether it concerns {c}:
a victim, organisation, government body, law, event, location or audience in {c} is part of what the post reports.
A post that merely lists {c} among many places, or matches only by an incidental word, counts as "no".
Use "unclear" only if the post cannot be judged.

Post:
<<<
{t}
>>>

Answer with one word: yes, no or unclear."""


def ask(client, model, text, country):
    body = {"model": model, "max_tokens": 5, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": PROMPT.format(c=NICE[country], t=text[:3000])}]}
    for attempt in range(5):
        try:
            r = client.post(os.environ["RELAY_BASE_URL"] + "/chat/completions", json=body,
                            headers={"Authorization": "Bearer " + os.environ["RELAY_API_KEY"]}, timeout=120)
            r.raise_for_status()
            a = r.json()["choices"][0]["message"]["content"].strip().lower()
            m = re.match(r"(yes|no|unclear)", a)
            return m.group(1) if m else "invalid:" + a[:20]
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
    return "error"


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    ids = [int(r["post_id"]) for r in csv.DictReader(open("manifests/interaction_experiment/www_scale/seed_manifest.csv"))]
    with db.connect() as c:
        txt = {r["post_id"]: r["text"] for r in c.execute("SELECT post_id, text FROM posts WHERE post_id = ANY(%s)", (ids,))}
    by = defaultdict(list)
    for p in sorted(txt):
        for k in countries(txt[p]):
            by[k].append(p)
    rng = random.Random(SEED)
    items = [(k, p) for k in COUNTRIES for p in rng.sample(by[k], min(PER, len(by[k])))]
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client() as client, ThreadPoolExecutor(8) as pool:
        lab = {role: list(pool.map(lambda kp: ask(client, m, txt[kp[1]], kp[0]), items)) for role, m in MODELS.items()}
    with open(OUT / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["country", "post_id", "reference_sonnet", "second_haiku"])
        for (k, p), a, b in zip(items, lab["reference"], lab["second"]):
            w.writerow([k, p, a, b])
    summ = {"seed": SEED, "per_country": PER, "models": MODELS, "note": "LLM labels; not human validation",
            "stratum_size_in_manifest": {k: len(by[k]) for k in COUNTRIES}, "countries": {}}
    for k in COUNTRIES:
        ref = [a for (kk, _), a in zip(items, lab["reference"]) if kk == k]
        sec = [b for (kk, _), b in zip(items, lab["second"]) if kk == k]
        dec = [a for a in ref if a in ("yes", "no")]
        summ["countries"][k] = {"n": len(ref), "reference": dict(Counter(ref)), "second": dict(Counter(sec)),
                                "precision_reference": round(dec.count("yes") / len(dec), 4) if dec else None,
                                "agreement": round(sum(a == b for a, b in zip(ref, sec)) / len(ref), 4)}
    allr, alls = lab["reference"], lab["second"]
    dec = [a for a in allr if a in ("yes", "no")]
    summ["overall"] = {"n": len(allr), "precision_reference": round(dec.count("yes") / len(dec), 4),
                       "agreement": round(sum(a == b for a, b in zip(allr, alls)) / len(allr), 4),
                       "kappa": round(kappa(allr, alls), 4)}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ["overall"]), json.dumps({k: (v["precision_reference"], v["agreement"]) for k, v in summ["countries"].items()}))


if __name__ == "__main__":
    main()
