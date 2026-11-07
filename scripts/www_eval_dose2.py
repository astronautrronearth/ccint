"""PREREG_DOSE2 evaluation: discordance vs engagement gain on six new domains (+ the five known settings).

    python scripts/www_eval_dose2.py → outputs/interaction_experiment/www_eval/dose2.json
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path[:0] = ["src", "scripts"]
from ccint import db  # noqa: E402
from ccint.interaction import manifest as mf  # noqa: E402

DOMS = ["climate", "film", "games", "soccer", "space", "stocks"]
FR = Path("www2027/results/frozen_eval")


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DOSE = _mod("dose", "scripts/www_eval_dose.py")
SC, CON, CC = DOSE.SC, DOSE.CON, DOSE.CC


def main():
    m = mf.load("www_dose2")
    with db.connect() as c:
        runs = [r["run_id"] for r in c.execute(
            "SELECT run_id FROM neighborhood_runs WHERE purpose = 'dose2' AND manifest_sha256 = %s", (m.sha256,))]
    pre = set(CON.HIST) | set(CON.DEV)
    new = {}
    for dom in DOMS:
        terms = getattr(CC, f"{dom.upper()}_TERMS")
        pat = {t: re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.I) for t in terms}
        unit_of = lambda r, pat=pat: [t for t, p in pat.items() if p.search(r["text"] or "")] or ["other"]
        ids = [int(r["post_id"]) for r in m.population(f"in_{dom}")]
        res = CON.domain(dom, ids, runs, unit_of)
        obs, _, info = SC.load(runs, ids)
        up_pre, up_all = defaultdict(list), defaultdict(list)
        for s in obs:
            for u in unit_of(info[s]):
                up_all[u].append(info[s]["author_id"])
                if info[s]["published_at"].date() in pre:
                    up_pre[u].append(info[s]["author_id"])
        C, ratio, nu = DOSE.concentration(up_pre)
        Call, ratio_all, _ = DOSE.concentration(up_all)
        k = res["monitor"]
        new[dom] = {"n": res["n_roots"], "n_observed": len(obs), "C_pre_eval": C, "C_all": Call, "units_used": nu,
                    "accounts_per_neff_pre": ratio, "accounts_per_neff_all": ratio_all,
                    "gain_edges": k["k3"]["P2_edges"]["vs_posts_rel"], "gain_edges_ci": k["k3"]["P2_edges"]["vs_posts_rel_ci"],
                    "gain_resp": k["k3"]["P4_responders"]["vs_posts_rel"], "posts_total": k["k3"]["P1_posts"]["total"],
                    "gain_edges_k1": k["k1"]["P2_edges"]["vs_posts_rel"], "gain_edges_k5": k["k5"]["P2_edges"]["vs_posts_rel"],
                    "persistent_gain": k["k3"]["P10_persistent"]["vs_posts_rel"],
                    "detail": {kk: res[kk] for kk in ("rho_posts_accounts", "rho_posts_neff", "self_share_48h",
                                                      "continuation_share", "incidence_48h", "units_ge30")}}
        print(dom, {kk: v for kk, v in new[dom].items() if kk != "detail"}, flush=True)
    sc = json.loads((FR / "story_checks.json").read_text())["discordance_pre_eval"]
    known = {k: {"C_pre_eval": sc[k]["C_pre_eval"], "gain_edges": sc[k]["gain_edges"]}
             for k in ("cyber_main_all", "cyber_fresh", "crypto", "ai", "nba")}
    ok = [d for d in DOMS if new[d]["C_pre_eval"] is not None and new[d]["gain_edges"] is not None]
    out = {"new": new, "known": known, "estimable_new": ok}
    if len(ok) >= 3:
        x = [new[d]["C_pre_eval"] for d in ok]
        y = [new[d]["gain_edges"] for d in ok]
        obs_r = spearmanr(x, y).statistic
        perms = [spearmanr(x, list(p)).statistic for p in permutations(y)]
        out["primary"] = {"rho": round(float(obs_r), 3), "p_one_sided": round(sum(r >= obs_r - 1e-12 for r in perms) / len(perms), 4),
                          "n": len(ok)}
        out["reading"] = "supported" if obs_r > 0 and out["primary"]["p_one_sided"] < 0.05 else "not supported"
        xs = sorted((v["C_pre_eval"], v["gain_edges"]) for v in known.values())
        pred = [float(np.interp(new[d]["C_pre_eval"], [a for a, _ in xs], [b for _, b in xs])) for d in ok]
        out["prediction"] = {"pred": dict(zip(ok, [round(p, 4) for p in pred])),
                             "rho_pred_obs": round(float(spearmanr(pred, y).statistic), 3)}
        allx = x + [v["C_pre_eval"] for v in known.values()]
        ally = y + [v["gain_edges"] for v in known.values()]
        out["all_eleven"] = DOSE.perm_test(allx, ally)
        out["secondary_C_all_new"] = round(float(spearmanr([new[d]["C_all"] for d in ok], y).statistic), 3)
        out["secondary_resp_new"] = round(float(spearmanr(x, [new[d]["gain_resp"] for d in ok]).statistic), 3)
    print(json.dumps({k: v for k, v in out.items() if k not in ("new", "known")}, indent=1))
    Path("outputs/interaction_experiment/www_eval/dose2.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
