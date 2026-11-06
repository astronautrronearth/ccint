"""Dose-response test: does producer concentration predict the gain of engagement over volume?
(PREREG: www2027/results/frozen_eval/PREREG_DOSE.md; deviations in DEVIATIONS.md)

    python scripts/www_eval_dose.py → outputs/interaction_experiment/www_eval/dose.json

Units: the larger cybersecurity collection replayed in full (E-A) and within each country stratum with >= 300
seeds (E-A protocol restricted to the stratum); the fresh cybersecurity window and NBA (E-D, contrast.json);
crypto and AI (same protocol as E-D on the www_contrast2 manifest). The PREREG listed the 'all' stratum both as
unit 1 and among the strata; it is counted once (12 units).
Concentration C = 1 - Spearman(posts, N_eff) over the unit's topics/terms with >= 30 posts (posts only).
Outcome: relative gain of ranking by edges (P2) over posts (P1), k = 3, evaluation days of the unit.
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


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SC = _mod("scale", "scripts/www_scale.py")
MON = _mod("mon", "scripts/www_eval_monitor.py")
CON = _mod("con", "scripts/www_eval_contrast.py")
CC = _mod("cc", "scripts/collect_contrast.py")
from country_filters import countries  # noqa: E402


def concentration(units_posts):
    """units_posts: {unit: [author_id, ...]} -> (C, median A/N_eff, n units used)."""
    rows = []
    for u, auth in units_posts.items():
        if len(auth) < 30:
            continue
        c = Counter(auth)
        n = len(auth)
        neff = 1 / sum((v / n) ** 2 for v in c.values())
        rows.append((n, len(c), neff))
    if len(rows) < 4:
        return None, None, len(rows)
    rho = spearmanr([r[0] for r in rows], [r[2] for r in rows]).statistic
    return round(float(1 - rho), 4), round(float(np.median([r[1] / r[2] for r in rows])), 3), len(rows)


def scale_units():
    m = mf.load("www_scale")
    ids = [int(r["post_id"]) for r in m.population("in_scale")]
    runs = SC.scale_runs(m.sha256)
    obs, edges, info = SC.load(runs, ids)
    seeds = sorted(s for s in obs if info[s]["topic"] != "no_topic")
    eb = defaultdict(list)
    for e in edges:
        if e["edge_type"] in ("reply", "quote") and e["seed_post_id"] in info:
            eb[e["seed_post_id"]].append(e)
    with db.connect() as c:
        th = {r["post_id"]: r["text_hash"] for r in c.execute("SELECT post_id, text_hash FROM posts WHERE post_id = ANY(%s)", (seeds,))}
    hist_ids = [s for s in ids if info.get(s) and info[s]["published_at"].date() in set(MON.HIST)]
    feeds = SC.feed_rules(hist_ids, info)["volume_only"]
    cs = {s: countries(info[s]["text"]) for s in seeds}
    strata = {"cyber_main_all": seeds, "cyber_main_no_country": [s for s in seeds if not cs[s]]}
    for k in ("united_states", "united_kingdom", "canada", "germany", "india", "australia"):
        strata[f"cyber_main_{k}"] = [s for s in seeds if k in cs[s]]
    out = {}
    for name, L in strata.items():
        cells, _ = MON.build(L, info, eb, th, feeds)
        topics = sorted({t for t, _ in cells})
        r = MON.evaluate(cells, topics, MON.EVAL, 3)
        up = defaultdict(list)
        for s in L:
            up[info[s]["topic"]].append(info[s]["author_id"])
        C, ratio, nu = concentration(up)
        out[name] = {"n": len(L), "C": C, "accounts_per_neff": ratio, "units_used": nu,
                     "gain_edges": r["P2_edges"]["vs_posts_rel"], "gain_edges_ci": r["P2_edges"]["vs_posts_rel_ci"],
                     "gain_resp": r["P4_responders"]["vs_posts_rel"], "posts_total": r["P1_posts"]["total"]}
        print(name, out[name], flush=True)
    return out


def contrast_units():
    """Fresh cyber and NBA from contrast.json; crypto and AI computed here with the E-D protocol."""
    out = {}
    d = json.loads(Path("outputs/interaction_experiment/www_eval/contrast.json").read_text())
    m = mf.load("www_contrast")
    runs = CON.runs_for(m.sha256)
    for dom, col in (("cyber", "in_cyber"), ("nba", "in_nba")):
        ids = [int(r["post_id"]) for r in m.population(col)]
        obs, _, info = SC.load(runs, ids)
        unit_of = (lambda r: [t for t, p in CON.PAT.items() if p.search(r["text"] or "")] or ["NBA"]) if dom == "nba" \
            else (lambda r: [r["topic"]] if r["topic"] != "no_topic" else [])
        up = defaultdict(list)
        for s in obs:
            for u in unit_of(info[s]):
                up[u].append(info[s]["author_id"])
        C, ratio, nu = concentration(up)
        k = d[dom]["monitor"]["k3"]
        out[f"{dom}_fresh" if dom == "cyber" else dom] = {"n": d[dom]["n_roots"], "C": C, "accounts_per_neff": ratio,
            "units_used": nu, "gain_edges": k["P2_edges"]["vs_posts_rel"], "gain_edges_ci": k["P2_edges"]["vs_posts_rel_ci"],
            "gain_resp": k["P4_responders"]["vs_posts_rel"], "posts_total": k["P1_posts"]["total"]}
    m2 = mf.load("www_contrast2")
    with db.connect() as c:
        runs2 = [r["run_id"] for r in c.execute(
            "SELECT run_id FROM neighborhood_runs WHERE purpose = 'contrast2' AND manifest_sha256 = %s", (m2.sha256,))]
    for dom, terms in (("crypto", CC.CRYPTO_TERMS), ("ai", CC.AI_TERMS)):
        pat = {t: re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in terms}
        ids = [int(r["post_id"]) for r in m2.population(f"in_{dom}")]
        unit_of = lambda r, pat=pat: [t for t, p in pat.items() if p.search(r["text"] or "")] or ["other"]
        res = CON.domain(dom, ids, runs2, unit_of)
        obs, _, info = SC.load(runs2, ids)
        up = defaultdict(list)
        for s in obs:
            for u in unit_of(info[s]):
                up[u].append(info[s]["author_id"])
        C, ratio, nu = concentration(up)
        k = res["monitor"]["k3"]
        out[dom] = {"n": res["n_roots"], "C": C, "accounts_per_neff": ratio, "units_used": nu,
                    "gain_edges": k["P2_edges"]["vs_posts_rel"], "gain_edges_ci": k["P2_edges"]["vs_posts_rel_ci"],
                    "gain_resp": k["P4_responders"]["vs_posts_rel"], "posts_total": k["P1_posts"]["total"],
                    "detail": {kk: res[kk] for kk in ("rho_posts_accounts", "rho_posts_neff", "self_share_24h",
                                                      "continuation_share", "incidence_24h")}}
        print(dom, out[dom], flush=True)
    return out


def perm_test(x, y, n=10000, seed=20261106):
    r = np.random.default_rng(seed)
    obs = spearmanr(x, y).statistic
    cnt = sum(spearmanr(x, r.permutation(y)).statistic >= obs for _ in range(n))
    return round(float(obs), 3), round((cnt + 1) / (n + 1), 4)


def main():
    units = {**scale_units(), **contrast_units()}
    ok = {k: v for k, v in units.items() if v["C"] is not None and v["gain_edges"] is not None}
    keys = sorted(ok)
    res = {"units": units, "n_units": len(ok)}
    res["primary"] = perm_test([ok[k]["C"] for k in keys], [ok[k]["gain_edges"] for k in keys])
    res["secondary_ratio"] = perm_test([ok[k]["accounts_per_neff"] for k in keys], [ok[k]["gain_edges"] for k in keys])
    res["secondary_resp"] = perm_test([ok[k]["C"] for k in keys], [ok[k]["gain_resp"] for k in keys])
    dom = [k for k in keys if not k.startswith("cyber_main_") or k == "cyber_main_all"]
    res["domains_only"] = {"units": dom, "rho_p": perm_test([ok[k]["C"] for k in dom], [ok[k]["gain_edges"] for k in dom])}
    res["reading"] = "supported" if res["primary"][0] > 0 and res["primary"][1] < 0.05 else "not supported"
    print(json.dumps({k: v for k, v in res.items() if k != "units"}, indent=1))
    Path("outputs/interaction_experiment/www_eval/dose.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
