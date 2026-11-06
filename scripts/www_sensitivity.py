"""Sensitivity analyses for the WWW short paper (brief §7, §17) and, once the manual audits exist,
the human-label versions (§6).

    python scripts/www_sensitivity.py                      → outputs/interaction_experiment/www_gonogo/sensitivity.json
    python scripts/www_sensitivity.py --audit outputs/www_audit
        reads canada/A.csv, feed/A.csv (primary, all items), optional B.csv (reliability subset) and adjudicated.csv
        and adds: Canada precision (Wilson CI), feed-rule accuracy, kappa, and the main statistics with
        human feed labels (accounts not in the audit, all with < 3 seeds and unflagged, count as non-feed).

Definitions match scripts/www_gonogo.py: observed main seeds, interaction = replies (depth <= 2) + quotes,
self edge = written by the seed author, CSR = d1 reply or quote, author-clustered bootstrap, seed-level
topic permutation. Output has no handles except for accounts whose actor_v1 type is news / org / intel feed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location("desc", "scripts/interaction_descriptive.py")
desc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(desc)
from ccint import db  # noqa: E402
from ccint.analytics.broadcast import SQL as BSQL, detect_broadcast_authors  # noqa: E402
from ccint.interaction.collect import EP_QUOTES, EP_THREAD  # noqa: E402

OK = {"success", "partial"}
PUBLIC = ("news_", "org_", "intel_feed")
rng = np.random.default_rng(0)


def load():
    m, runs, seeds, oc, edges, info, tc = desc.load()
    obs = lambda s: bool(oc.get(s)) and oc[s][EP_THREAD]["outcome"] in OK and oc[s][EP_QUOTES]["outcome"] in OK
    partial = {s for s in seeds if obs(s) and oc[s][EP_THREAD]["outcome"] == "partial"}
    pops = {"main": [s for s, r in seeds.items() if r["in_main"] == "true" and obs(s)],
            "legacy_llm_v2c": [s for s, r in seeds.items() if r["in_legacy"] == "true" and obs(s)]}
    ps = defaultdict(Counter)
    for e in edges:
        s = e["seed_post_id"]
        if e["edge_type"] not in ("reply", "quote"):
            continue
        self_ = e["source_actor_id"] == info[s]["author_id"]
        k = "q" if e["edge_type"] == "quote" else ("d1" if e["depth"] == 1 else "d2")
        ps[s][k] += 1
        ps[s][k + "_ns"] += not self_
    with db.connect() as c:
        beh = detect_broadcast_authors(c, label_version="llm_v2d").author_ids
        vol = {r["author_id"] for r in c.execute(BSQL, {"label_version": "llm_v2d", "since": None, "until": None,
               "min_lag_hours": 24}) if r["n_posts"] >= 5 and r["n_days"] >= 4}
    return pops, partial, ps, info, beh, vol


def inter(ps, s, ns):
    k = "_ns" if ns else ""
    return ps[s]["d1" + k] + ps[s]["d2" + k] + ps[s]["q" + k]


def conv(ps, s, ns):
    k = "_ns" if ns else ""
    return ps[s]["d1" + k] > 0 or ps[s]["q" + k] > 0


def boot(ids, author, f, B=4000):
    by = defaultdict(list)
    for s in ids:
        by[author[s]].append(f(s))
    g = list(by.values())
    num = np.array([sum(x) for x in g], float)
    den = np.array([len(x) for x in g], float)
    idx = rng.integers(0, len(g), (B, len(g)))
    return [round(float(x), 4) for x in np.percentile(num[idx].sum(1) / den[idx].sum(1), [2.5, 97.5])]


def perm(ids, topic, y, B=5000):
    t = np.array([topic[s] for s in ids])
    ks = sorted(set(t))
    T = np.array([ks.index(x) for x in t])
    n = np.bincount(T, minlength=len(ks))
    yy = np.array([y(s) for s in ids], float)
    g = yy.mean()
    if g == 0:
        return None
    D = lambda TT: (n * (np.bincount(TT, weights=yy, minlength=len(ks)) / n - g) ** 2).sum() / g
    d0 = D(T)
    return float((1 + sum(D(rng.permutation(T)) >= d0 for _ in range(B))) / (B + 1))


def summary(ids, ps, info, feeds=None):
    """Main statistics on a seed list; feeds = set of author ids to remove first (None = keep all)."""
    author = {s: info[s]["author_id"] for s in ids}
    topic = {s: info[s]["topic"] for s in ids}
    if feeds is not None:
        ids = [s for s in ids if author[s] not in feeds]
    out = {"n": len(ids)}
    tot, ns = sum(inter(ps, s, False) for s in ids), sum(inter(ps, s, True) for s in ids)
    out["edges"], out["self_edges"] = tot, tot - ns
    out["self_share"] = round((tot - ns) / tot, 4) if tot else None
    for lab, flag in (("raw", False), ("nonself", True)):
        out[f"CSR_{lab}"] = round(float(np.mean([conv(ps, s, flag) for s in ids])), 4)
        out[f"CSR_{lab}_ci"] = boot(ids, author, lambda s: conv(ps, s, flag))
        out[f"perm_p_{lab}"] = perm(ids, topic, lambda s: inter(ps, s, flag))
    T = defaultdict(list)
    for s in ids:
        T[topic[s]].append(s)
    rows = {k: {"posts": len(L), "inter_raw": sum(inter(ps, s, False) for s in L),
                "inter_nonself": sum(inter(ps, s, True) for s in L)} for k, L in T.items()}
    for col in ("posts", "inter_raw", "inter_nonself"):
        for r, k in enumerate(sorted(rows, key=lambda k: (-rows[k][col], k)), 1):
            rows[k][f"rank_{col}"] = r
    for k in rows:
        rows[k]["per_post_nonself"] = round(rows[k]["inter_nonself"] / rows[k]["posts"], 3)
    out["topics"] = rows
    if "ransomware" in T:
        rw = T["ransomware"]
        rest = [s for s in ids if topic[s] != "ransomware"]
        r_rw = sum(inter(ps, s, True) for s in rw) / len(rw)
        r_rest = sum(inter(ps, s, True) for s in rest) / len(rest)
        out["ransomware"] = {"posts": len(rw), "self_edges": sum(inter(ps, s, False) - inter(ps, s, True) for s in rw),
                             "edges": sum(inter(ps, s, False) for s in rw),
                             "per_post_nonself": round(r_rw, 3), "ratio_vs_rest": round(r_rw / r_rest, 3) if r_rest else None}
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def kappa(a, b):
    labs = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labs)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def read_export(p):
    return {r["item_id"]: r for r in csv.DictReader(open(p))}


def labels(adir, task, field):
    """Primary labels = annotator A on every item; B on a reliability subset. If adjudicated.csv exists
    (item_id,<field>) it overrides A on the items it lists (disagreements resolved after kappa)."""
    A, B = read_export(adir / f"{task}/A.csv"), {}
    if (adir / f"{task}/B.csv").exists():
        B = read_export(adir / f"{task}/B.csv")
    prim = {i: r[field] for i, r in A.items() if r.get(field)}
    adj = adir / f"{task}/adjudicated.csv"
    n_adj = 0
    if adj.exists():
        for i, r in read_export(adj).items():
            if r.get(field):
                prim[i] = r[field]
                n_adj += 1
    both = [i for i in A if A[i].get(field) and B.get(i, {}).get(field)]
    a, b = [A[i][field] for i in both], [B[i][field] for i in both]
    rel = {"double_coded": len(both), "raw_agreement": round(float(np.mean([x == y for x, y in zip(a, b)])), 3) if both else None,
           "kappa": kappa(a, b) if both else None,
           "disagreements": [i for i, x, y in zip(both, a, b) if x != y], "adjudicated_items": n_adj}
    return prim, rel, A, B, both


def audit(adir, pops, ps, info, beh, vol):
    out = {}
    # Canada: precision of the llm_v2d filter = share of yes among decided primary labels
    prim, rel, *_ = labels(adir, "canada", "canada_cyber")
    c = Counter(prim.values())
    dec = c["yes"] + c["no"]
    out["canada"] = {"n_primary": len(prim), "counts": dict(c), "unclear_rate": round(c["unclear"] / len(prim), 4),
                     "precision": round(c["yes"] / dec, 4) if dec else None, "precision_ci": wilson(c["yes"], dec),
                     "reliability": rel}
    from strict_canada_filter import strict_canada
    key = {r["item_id"]: r for r in csv.DictReader(open(adir / "canada/key.csv"))}
    with db.connect() as cn:
        txt = {r["post_id"]: r["text"] for r in cn.execute("SELECT post_id, text FROM posts WHERE post_id = ANY(%s)",
                                                         ([int(key[i]["post_id"]) for i in prim],))}
    cm = Counter((strict_canada(txt[int(key[i]["post_id"])]), v) for i, v in prim.items() if v != "unclear")
    tp, fp, fn = cm[(True, "yes")], cm[(True, "no")], cm[(False, "yes")]
    out["canada"]["strict_filter"] = {"precision": round(tp / (tp + fp), 4) if tp + fp else None,
                                      "precision_ci": wilson(tp, tp + fp), "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                                      "tp": tp, "fp": fp, "fn": fn}
    # Feed
    prim, rel, A, B, both = labels(adir, "feed", "feed")
    key = {r["item_id"]: r for r in csv.DictReader(open(adir / "feed/key.csv"))}
    rel["kappa_role"] = kappa([A[i].get("role", "") for i in both], [B[i].get("role", "") for i in both]) if both else None
    human = {key[i]["author_id"]: v for i, v in prim.items()}
    res = {"n_primary": len(human), "counts": dict(Counter(human.values())), "reliability": rel, "rules": {}}
    for name, F in (("behavioral_v1", beh), ("volume_only", vol)):
        cm = Counter((int(aid in F), v) for aid, v in human.items())
        tp, fp, fn = cm[(1, "feed")], cm[(1, "non_feed")], cm[(0, "feed")]
        res["rules"][name] = {"confusion(rule_flag,human)": {f"{k[0]},{k[1]}": v for k, v in sorted(cm.items())},
                              "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                              "recall": round(tp / (tp + fn), 3) if tp + fn else None}
    out["feed"] = res
    hfeed = {a for a, v in human.items() if v == "feed"}
    out["main_with_human_feeds"] = summary(pops["main"], ps, info, hfeed)
    with db.connect() as cn:
        txt = {r["post_id"]: r["text"] for r in cn.execute("SELECT post_id, text FROM posts WHERE post_id = ANY(%s)", (pops["main"],))}
    out["strict_canada_with_human_feeds"] = summary([s for s in pops["main"] if strict_canada(txt[s])], ps, info, hfeed)
    out["main_with_human_feeds_or_unclear"] = summary(pops["main"], ps, info,
                                                      hfeed | {a for a, v in human.items() if v == "unclear"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path)
    args = ap.parse_args()
    pops, partial, ps, info, beh, vol = load()
    M = pops["main"]
    res = {"n_partial_main": len(set(M) & partial)}
    prod = Counter(info[s]["author_id"] for s in M)
    ranked = [a for a, _ in prod.most_common()]
    res["variants"] = {
        "main": summary(M, ps, info),
        "main_no_partial": summary([s for s in M if s not in partial], ps, info),
        **{f"main_minus_top{k}": summary(M, ps, info, set(ranked[:k])) for k in (1, 5, 10)},
        "legacy_llm_v2c": summary(pops["legacy_llm_v2c"], ps, info),
        "feeds_behavioral": summary(M, ps, info, beh),
        "feeds_volume_only": summary(M, ps, info, vol),
        "feeds_behavioral_no_partial": summary([s for s in M if s not in partial], ps, info, beh),
    }
    # replication on the strict Canada subset (scripts/strict_canada_filter.py; decided 2026-11-05)
    from strict_canada_filter import strict_canada
    with db.connect() as c:
        txt = {r["post_id"]: r["text"] for r in c.execute("SELECT post_id, text FROM posts WHERE post_id = ANY(%s)", (M,))}
    SC = [s for s in M if strict_canada(txt[s])]
    res["strict_canada_n"] = len(SC)
    res["variants"].update({"strict_canada": summary(SC, ps, info),
                            "strict_canada_feeds_behavioral": summary(SC, ps, info, beh),
                            "strict_canada_feeds_volume_only": summary(SC, ps, info, vol)})
    # rank changes (Q7): feed removal on post rank; self removal on interaction rank; both
    base, fb, fv = res["variants"]["main"]["topics"], res["variants"]["feeds_behavioral"]["topics"], \
        res["variants"]["feeds_volume_only"]["topics"]
    res["rank_changes"] = {k: {"post_rank": base[k]["rank_posts"], "post_rank_feeds_beh": fb[k]["rank_posts"],
                               "post_rank_feeds_vol": fv[k]["rank_posts"],
                               "inter_rank_raw": base[k]["rank_inter_raw"], "inter_rank_nonself": base[k]["rank_inter_nonself"],
                               "inter_rank_both_beh": fb[k]["rank_inter_nonself"], "inter_rank_both_vol": fv[k]["rank_inter_nonself"]}
                           for k in base}
    # Q3: accounts flagged by one rule only
    def label(a):
        t = next((info[s]["actor_type"] for s in M if info[s]["author_id"] == a), "")
        h = next(info[s]["author_handle"] for s in M if info[s]["author_id"] == a)
        return h if t.startswith(PUBLIC) else "anon:" + hashlib.sha256(a.encode()).hexdigest()[:8], t
    dis = []
    for a in sorted(vol ^ beh, key=lambda a: -prod.get(a, 0)):
        if prod.get(a):
            h, t = label(a)
            tops = Counter(info[s]["topic"] for s in M if info[s]["author_id"] == a).most_common(2)
            dis.append({"account": h, "actor_type_v1": t, "rule": "volume_only" if a in vol else "behavioral_v1",
                        "main_seeds": prod[a], "top_topics": tops})
    res["rule_disagreement_accounts"] = dis
    if args.audit:
        res["audit"] = audit(args.audit, pops, ps, info, beh, vol)
    p = Path("outputs/interaction_experiment/www_gonogo/sensitivity.json")
    p.write_text(json.dumps(res, indent=1, default=str))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
