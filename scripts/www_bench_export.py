"""Export the topic-selection benchmark (ccbench v1) to DB-free JSON files (2026-11-08).

One file per setting with, per (topic, day) cell, everything a method may observe at the daily cutoff and the
target, built exactly as scripts/www_manip_adaptive.py (cells_of) and scripts/www_budget_fetch.py (prepare) do:
  posts            [author, replies+quotes before the cutoff] per seed post, in collection order
  responders       {account: replies+quotes before the cutoff} by external accounts
  conversational   accounts that replied to a reply or whose reply drew a reply (before the cutoff)
  answered_by      {depth-1 replier: [accounts that answered that reply]} (before the cutoff)
  unanswered       depth-1 repliers whose reply was not answered (before the cutoff)
  target           external accounts that first engage in the 24 h after the cutoff (not before it)
Accounts are pseudonymous integers: the rank of the internal account id in one global sorted list (no DIDs, no
handles, no text, no URIs). The rank preserves the original order, so tie-breaks and the attack simulation
reproduce exactly. Topics are the public search terms / topic labels.

    python scripts/www_bench_export.py → outputs/ccbench_v1/<setting>.json, splits.json
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("ma", "scripts/www_manip_adaptive.py")
MA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MA)
spec2 = importlib.util.spec_from_file_location("bf", "scripts/www_budget_fetch.py")
BF = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(BF)
BF.ECAP = 10 ** 9
TIER = {**{n: "dev" for n in ["cyber_main", "cyber_fresh", "nba", "crypto", "ai"]},
        **{n: "held" for n in ["climate", "film", "games", "soccer", "space", "stocks"]},
        **{n: "test" for n in ["music", "books", "f1", "tennis", "health", "food"]}}
OUT = Path("outputs/ccbench_v1")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SS = MA.MECH.settings(False)
    built = {}
    ids = set()
    for n in TIER:
        S = SS[n]
        cells = MA.cells_of(S)
        cb = BF.prepare(S, {}, {})
        rows = []
        for (t, d) in cells:   # keep the original insertion order (victim tie-breaks depend on it)
            c, b = cells[(t, d)], cb[(t, d)]
            authors = [S["info"][s]["author_id"] for s in b["posts"]]
            assert len(authors) == c["V"] and len(set(authors)) == c["A"] and abs(sum(b["e"]) - c["E"]) < 1e-9
            row = dict(topic=t, day=str(d), posts=[[a, int(k)] for a, k in zip(authors, b["e"])],
                       responders=dict(c["resp"]), conversational=sorted(c["conv"]),
                       answered_by={j: sorted(v) for j, v in c["recv_by"].items()}, unanswered=sorted(c["d1u"]),
                       target=sorted(c["T"]))
            ids.update(authors, c["resp"], c["conv"], c["recv_by"], c["d1u"], c["T"])
            for v in c["recv_by"].values():
                ids.update(v)
            rows.append(row)
        built[n] = (rows, S)
        print("built", n, len(rows), flush=True)
    rank = {a: i for i, a in enumerate(sorted(ids))}
    R = lambda a: rank[a]
    for n, (rows, S) in built.items():
        for r in rows:
            r["posts"] = [[R(a), k] for a, k in r["posts"]]
            r["responders"] = {str(R(a)): k for a, k in r["responders"].items()}
            r["conversational"] = [R(a) for a in r["conversational"]]
            r["answered_by"] = {str(R(j)): [R(a) for a in v] for j, v in r["answered_by"].items()}
            r["unanswered"] = [R(a) for a in r["unanswered"]]
            r["target"] = [R(a) for a in r["target"]]
        doc = dict(setting=n, tier=TIER[n], k=3, days={s: [str(x) for x in sorted(S[s])] for s in ("hist", "dev", "eval")},
                   cells=rows)
        (OUT / f"{n}.json").write_text(json.dumps(doc))
    (OUT / "splits.json").write_text(json.dumps({t: [n for n in TIER if TIER[n] == t] for t in ("dev", "held", "test")}, indent=1))
    print("accounts", len(rank))


if __name__ == "__main__":
    main()
