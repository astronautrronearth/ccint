"""PREREG_DOSE2 amendment 1: topic-level mechanism (M), authors baseline (B), concentration-reducing intervention (I),
decision utility (U) and confound checks, over all available domain-window settings.

    python scripts/www_dose2_mech.py [--known-only] → outputs/interaction_experiment/www_eval/dose2_mech.json
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path[:0] = ["src", "scripts"]
from ccint import db  # noqa: E402
from ccint.interaction import manifest as mf  # noqa: E402

NEW = ["climate", "film", "games", "soccer", "space", "stocks"]


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DOSE = _mod("dose", "scripts/www_eval_dose.py")
SC, MON, CON, CC = DOSE.SC, DOSE.MON, DOSE.CON, DOSE.CC


def capture_build(fn):
    """Run fn() while capturing the arguments of CON.MON.build (pairs, pinfo, peb, pth, feeds)."""
    cap = {}
    orig = CON.MON.build

    def b(*a, **k):
        cap["args"] = a
        return orig(*a, **k)
    CON.MON.build = b
    try:
        fn()
    finally:
        CON.MON.build = orig
    return cap["args"]


def settings(known_only):
    """name -> dict(seeds, info, eb, th, feeds, hist, dev, eval)."""
    out = {}
    # cyber main (E-A protocol)
    m = mf.load("www_scale")
    ids = [int(r["post_id"]) for r in m.population("in_scale")]
    obs, edges, info = SC.load(SC.scale_runs(m.sha256), ids)
    seeds = sorted(s for s in obs if info[s]["topic"] != "no_topic")
    eb = defaultdict(list)
    for e in edges:
        if e["edge_type"] in ("reply", "quote") and e["seed_post_id"] in info:
            eb[e["seed_post_id"]].append(e)
    with db.connect() as c:
        th = {r["post_id"]: r["text_hash"] for r in c.execute("SELECT post_id, text_hash FROM posts WHERE post_id = ANY(%s)", (seeds,))}
    hist_ids = [s for s in ids if info.get(s) and info[s]["published_at"].date() in set(MON.HIST)]
    feeds = SC.feed_rules(hist_ids, info)["volume_only"]
    out["cyber_main"] = dict(seeds=seeds, info=info, eb=eb, th=th, feeds=feeds, hist=MON.HIST, dev=MON.DEV, eval=MON.EVAL)

    def add(name, dom, ids, runs, unit_of):
        args = capture_build(lambda: CON.domain(dom, ids, runs, unit_of))
        pairs, pinfo, peb, pth, feeds_ = args
        out[name] = dict(seeds=pairs, info=pinfo, eb=peb, th=pth, feeds=feeds_, hist=CON.HIST, dev=CON.DEV, eval=CON.EVAL)

    m = mf.load("www_contrast")
    runs = CON.runs_for(m.sha256)
    add("cyber_fresh", "cyber", [int(r["post_id"]) for r in m.population("in_cyber")], runs,
        lambda r: [r["topic"]] if r["topic"] != "no_topic" else [])
    add("nba", "nba", [int(r["post_id"]) for r in m.population("in_nba")], runs,
        lambda r: [t for t, p in CON.PAT.items() if p.search(r["text"] or "")] or ["NBA"])
    m2 = mf.load("www_contrast2")
    with db.connect() as c:
        runs2 = [r["run_id"] for r in c.execute(
            "SELECT run_id FROM neighborhood_runs WHERE purpose = 'contrast2' AND manifest_sha256 = %s", (m2.sha256,))]
    for dom, terms in (("crypto", CC.CRYPTO_TERMS), ("ai", CC.AI_TERMS)):
        pat = {t: re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in terms}
        add(dom, dom, [int(r["post_id"]) for r in m2.population(f"in_{dom}")], runs2,
            lambda r, pat=pat: [t for t, p in pat.items() if p.search(r["text"] or "")] or ["other"])
    if not known_only:
        m3 = mf.load("www_dose2")
        with db.connect() as c:
            runs3 = [r["run_id"] for r in c.execute(
                "SELECT run_id FROM neighborhood_runs WHERE purpose = 'dose2' AND manifest_sha256 = %s", (m3.sha256,))]
        for dom in NEW:
            terms = getattr(CC, f"{dom.upper()}_TERMS")
            pat = {t: re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.I) for t in terms}
            add(dom, dom, [int(r["post_id"]) for r in m3.population(f"in_{dom}")], runs3,
                lambda r, pat=pat: [t for t, p in pat.items() if p.search(r["text"] or "")] or ["other"])
    return out


def conc_by_topic(S, days):
    by = defaultdict(list)
    for s in S["seeds"]:
        r = S["info"][s]
        if r["published_at"].date() in days:
            by[r["topic"]].append(r["author_id"])
    q = {}
    for t, L in by.items():
        if len(L) >= 30:
            c = Counter(L)
            n = len(L)
            q[t] = len(c) / (1 / sum((v / n) ** 2 for v in c.values()))
    return q, by


def C_of(by):
    rows = []
    for t, L in by.items():
        if len(L) >= 30:
            c = Counter(L)
            n = len(L)
            rows.append((n, 1 / sum((v / n) ** 2 for v in c.values()), len({1})))
    if len(rows) < 4:
        return None
    return float(1 - spearmanr([r[0] for r in rows], [r[1] for r in rows]).statistic)


def ranks(v):
    from scipy.stats import rankdata
    return rankdata([-x for x in v], method="average")


def analyse(name, S, rng):
    pre = set(S["hist"]) | set(S["dev"])
    cells, _ = MON.build(S["seeds"], S["info"], S["eb"], S["th"], S["feeds"])
    MON.add_persistent(cells, S["dev"])
    topics = sorted({t for t, _ in cells})
    ev = MON.evaluate(cells, topics, S["eval"], 3)
    q, by_pre = conc_by_topic(S, pre)
    res = {"gain_edges": ev["P2_edges"]["vs_posts_rel"], "gain_authors": ev["P3_authors"]["vs_posts_rel"],
           "edges_over_authors": (ev["P2_edges"]["total"] / ev["P3_authors"]["total"] - 1) if ev["P3_authors"]["total"] else None,
           "totals": {p: ev[p]["total"] for p in ("P1_posts", "P2_edges", "P3_authors", "P4_responders")},
           "C_pre": C_of(by_pre)}
    # M: topic-day over-ranking vs log concentration
    pts = []
    for d in S["eval"]:
        T = [t for t in topics if (t, d) in cells and cells[(t, d)]["P1_posts"] > 0]
        if len(T) < 2:
            continue
        rp = ranks([cells[(t, d)]["P1_posts"] for t in T])
        rt = ranks([cells[(t, d)]["target"] for t in T])
        for t, a, b in zip(T, rp, rt):
            if t in q:
                pts.append((str(d), t, float(np.log(q[t])), float(b - a), float(np.log(len(by_pre[t])))))
    res["M_points"] = pts
    # C bootstrap over pre-evaluation days; placebo C'
    pdays = sorted(pre)
    cb = []
    for _ in range(200):
        dd = [pdays[i] for i in rng.integers(0, len(pdays), len(pdays))]
        by = defaultdict(list)
        cnt = Counter(dd)
        for s in S["seeds"]:
            r = S["info"][s]
            k = cnt.get(r["published_at"].date(), 0)
            by[r["topic"]] += [r["author_id"]] * k
        v = C_of(by)
        if v is not None:
            cb.append(v)
    res["C_pre_ci"] = [float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))] if cb else None
    act = defaultdict(set)
    for s in S["seeds"]:
        r = S["info"][s]
        if r["published_at"].date() in pre:
            act[r["topic"]].add((r["published_at"].date(), r["published_at"].hour))
    big = [t for t, L in by_pre.items() if len(L) >= 30]
    res["C_placebo"] = float(1 - spearmanr([len(by_pre[t]) for t in big], [len(act[t]) for t in big]).statistic) if len(big) >= 4 else None
    # I: remove top-m output accounts' evaluation-day posts from ranking inputs (target fixed)
    evs = [s for s in S["seeds"] if S["info"][s]["published_at"].date() in set(S["eval"])]
    out_cnt = Counter(S["info"][s]["author_id"] for s in evs)
    top = [a for a, _ in out_cnt.most_common()]

    def gain_after(keep):
        kc, _ = MON.build(keep + [s for s in S["seeds"] if S["info"][s]["published_at"].date() not in set(S["eval"])],
                          S["info"], S["eb"], S["th"], S["feeds"])
        for key in kc:
            if key in cells:
                kc[key]["target"] = cells[key]["target"]
                kc[key]["P9_oracle"] = cells[key]["target"]
        for key in cells:                       # topic-days emptied by the removal keep their target
            kc.setdefault(key, {**{p: 0 for p in MON.POLICIES}, "target": cells[key]["target"]})
        g = {p: sum(MON.select_value(kc, topics, d, p, 3, 20) for d in S["eval"]) for p in ("P1_posts", "P2_edges")}
        return g["P2_edges"] / g["P1_posts"] - 1 if g["P1_posts"] else None
    I = {"0": res["gain_edges"]}
    for mm in (1, 3, 10):
        drop = set(top[:mm])
        keep = [s for s in evs if S["info"][s]["author_id"] not in drop]
        nrem = len(evs) - len(keep)
        rr = []
        for _ in range(20):
            idx = set(rng.choice(len(evs), size=nrem, replace=False)) if nrem else set()
            rr.append(gain_after([s for i, s in enumerate(evs) if i not in idx]))
        rr = [x for x in rr if x is not None]
        I[str(mm)] = {"removed_posts": nrem, "gain_top": gain_after(keep), "gain_random_mean": float(np.mean(rr)) if rr else None}
    res["I"] = I
    res["cost_eval_posts"] = len(evs)
    res["feed_share_eval"] = sum(S["info"][s]["author_id"] in S["feeds"] for s in evs) / max(1, len(evs))
    print(name, {k: v for k, v in res.items() if k != "M_points"}, flush=True)
    return res


def main(known_only):
    rng = np.random.default_rng(20261107)
    S = settings(known_only)
    R = {n: analyse(n, s, rng) for n, s in S.items()}
    out = {"settings": {n: {k: v for k, v in r.items() if k != "M_points"} for n, r in R.items()}}
    # M regression with setting fixed effects (demean within setting), bootstrap over setting-days
    rows = [(n, d, x, o) for n, r in R.items() for d, t, x, o, _ in r["M_points"]]
    # amendment 2: residualise log q on log size within setting
    rows_sz = []
    for n, r in R.items():
        P = r["M_points"]
        if len(P) < 3:
            continue
        X = np.array([[1, p[4]] for p in P])
        y = np.array([p[2] for p in P])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        rows_sz += [(n, p[0], float(p[2] - X[i] @ beta), p[3]) for i, p in enumerate(P)]

    def slope(rs):
        by = defaultdict(list)
        for n, d, x, o in rs:
            by[n].append((x, o))
        xs, ys = [], []
        for L in by.values():
            mx, my = np.mean([a for a, _ in L]), np.mean([b for _, b in L])
            xs += [a - mx for a, _ in L]
            ys += [b - my for _, b in L]
        xs, ys = np.array(xs), np.array(ys)
        return float((xs * ys).sum() / (xs ** 2).sum()) if (xs ** 2).sum() else None
    groups = defaultdict(list)
    for r in rows:
        groups[(r[0], r[1])].append(r)
    keys = list(groups)
    bs = []
    for _ in range(2000):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        v = slope([r for k in pick for r in groups[k]])
        if v is not None:
            bs.append(v)
    out["M"] = {"n_points": len(rows), "slope": slope(rows), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "slope_without_cyber": slope([r for r in rows if not r[0].startswith("cyber")]),
                "slope_by_setting": {n: slope([r for r in rows if r[0] == n]) for n in R},
                "slope_size_controlled": slope(rows_sz),
                "slope_size_controlled_by_setting": {n: slope([r for r in rows_sz if r[0] == n]) for n in R}}
    # B
    eoa = [r["edges_over_authors"] for r in R.values() if r["edges_over_authors"] is not None]
    out["B"] = {"median_edges_over_authors": float(np.median(eoa)),
                "rho_C_edges_over_authors": float(spearmanr([R[n]["C_pre"] for n in R if R[n]["C_pre"] is not None],
                                                            [R[n]["edges_over_authors"] for n in R if R[n]["C_pre"] is not None]).statistic)}
    # I sign tests
    ok = [n for n in R if R[n]["I"]["10"]["gain_top"] is not None and R[n]["gain_edges"] is not None]
    out["I"] = {"falls": sum(R[n]["I"]["10"]["gain_top"] < R[n]["gain_edges"] for n in ok),
                "falls_more_than_random": sum(R[n]["I"]["10"]["gain_top"] < (R[n]["I"]["10"]["gain_random_mean"] or 9) for n in ok),
                "n": len(ok)}
    # U: leave-one-setting-out gating on C (cyber windows held out together)
    names = [n for n in R if R[n]["C_pre"] is not None]
    gain_over_auth = {n: R[n]["totals"]["P2_edges"] - R[n]["totals"]["P3_authors"] for n in names}
    cost = {n: R[n]["cost_eval_posts"] for n in names}
    held = []
    for n in names:
        grp = [x for x in names if x.startswith("cyber")] if n.startswith("cyber") else [n]
        train = [x for x in names if x not in grp]
        lam = sum(gain_over_auth[x] for x in train) / sum(cost[x] for x in train)
        best = max(sorted({R[x]["C_pre"] for x in train}) + [9.0],
                   key=lambda tau: sum(gain_over_auth[x] - lam * cost[x] for x in train if R[x]["C_pre"] > tau))
        held.append((n, R[n]["C_pre"] > best, best))
    fetched = [n for n, f, _ in held if f]
    out["U"] = {"decisions": held, "gain_kept": sum(gain_over_auth[n] for n in fetched) / max(1e-9, sum(gain_over_auth.values())),
                "cost_share": sum(cost[n] for n in fetched) / sum(cost.values())}
    # confounds (descriptive)
    out["confounds"] = {k: float(spearmanr([R[n][kk] if kk != "logposts" else np.log(R[n]["totals"]["P1_posts"] + 1) for n in names],
                                          [R[n]["gain_edges"] for n in names]).statistic)
                        for k, kk in (("feed_share", "feed_share_eval"), ("log_posts_baseline", "logposts"), ("C_placebo", "C_placebo"))
                        if all(R[n].get(kk) is not None or kk == "logposts" for n in names)}
    print(json.dumps({k: v for k, v in out.items() if k != "settings"}, indent=1, default=str))
    tag = "_known" if known_only else ""
    Path(f"outputs/interaction_experiment/www_eval/dose2_mech{tag}.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main("--known-only" in sys.argv)
