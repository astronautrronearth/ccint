"""Producer decomposition for the WWW short paper (2026-11-06): producer-standardised interaction rates,
a Kitagawa decomposition of the ransomware gap, and producer concentration per topic.

    python scripts/www_producer.py --audit <scoring dir>   → outputs/interaction_experiment/www_gonogo/producer.json

Notation per topic t. Volume V_t = V_t^F + V_t^H (seeds by human-audited feed accounts / others).
Interaction I_t = I_t^self + I_t^cross (reply+quote edges by the seed author / by anyone else).
Cross-user rate r_t = I_t^cross / V_t = s_t r_tF + (1 - s_t) r_tH, with s_t the topic's feed share.
Standardised rate r~_t = s r_tF + (1 - s) r_tH with one common feed share s (the pooled share), i.e. every
topic given the same producer composition. Dispersion D = sum_t V_t (r_t - rbar)^2 / sum_t V_t.
Concentration: N_eff = 1 / sum_i p_i^2 over seed authors' post shares (effective number of producers).
CIs: author-clustered bootstrap (authors resampled with all their seeds), B = 2000.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("sens", "scripts/www_sensitivity.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
from ccint import db  # noqa: E402
from strict_canada_filter import strict_canada  # noqa: E402

B = 2000
rng = np.random.default_rng(7)


def human_feeds(adir):
    key = {r["item_id"]: r for r in csv.DictReader(open(adir / "feed/key.csv"))}
    lab = {r["item_id"]: r["feed"] for r in csv.DictReader(open(adir / "feed/A.csv")) if r["feed"]}
    adj = adir / "feed/adjudicated.csv"
    if adj.exists():
        lab.update({r["item_id"]: r["feed"] for r in csv.DictReader(open(adj)) if r["feed"]})
    return {key[i]["author_id"] for i, v in lab.items() if v == "feed"}


def stats(seeds, topic, author, feed, cross, topics):
    """Point estimates on a list of seeds (may contain repeats from the bootstrap)."""
    n = Counter(topic[s] for s in seeds)
    nF = Counter(topic[s] for s in seeds if feed[s])
    cx = Counter()
    cxF = Counter()
    for s in seeds:
        cx[topic[s]] += cross[s]
        if feed[s]:
            cxF[topic[s]] += cross[s]
    N, NF = sum(n.values()), sum(nF.values())
    sbar = NF / N
    out = {}
    for t in topics:
        if not n[t]:
            continue
        s_t = nF[t] / n[t]
        r = cx[t] / n[t]
        rF = cxF[t] / nF[t] if nF[t] else None
        rH = (cx[t] - cxF[t]) / (n[t] - nF[t]) if n[t] - nF[t] else None
        if rF is None:
            rF = rH
        if rH is None:
            rH = rF
        out[t] = {"n": n[t], "feed_share": s_t, "r": r, "rF": rF, "rH": rH, "r_std": sbar * rF + (1 - sbar) * rH}
    w = np.array([out[t]["n"] for t in out], float)
    def disp(k):
        x = np.array([out[t][k] for t in out])
        m = (w * x).sum() / w.sum()
        return float((w * (x - m) ** 2).sum() / w.sum())
    return out, {"D_raw": disp("r"), "D_std": disp("r_std"), "D_nonfeed": disp("rH"), "pooled_feed_share": sbar}


def kitagawa(o, t, rest):
    """Gap r_t - r_rest = composition + within, with feed/non-feed as the two producer groups."""
    a, b = o[t], rest
    sa, sb = a["feed_share"], b["feed_share"]
    comp = (sa - sb) * (a["rF"] + b["rF"]) / 2 + ((1 - sa) - (1 - sb)) * (a["rH"] + b["rH"]) / 2
    within = (a["rF"] - b["rF"]) * (sa + sb) / 2 + (a["rH"] - b["rH"]) * ((1 - sa) + (1 - sb)) / 2
    return {"gap": a["r"] - b["r"], "composition": comp, "within": within}


def rest_of(seeds, topic, feed, cross, t):
    L = [s for s in seeds if topic[s] != t]
    F = [s for s in L if feed[s]]
    H = [s for s in L if not feed[s]]
    return {"feed_share": len(F) / len(L), "r": sum(cross[s] for s in L) / len(L),
            "rF": sum(cross[s] for s in F) / len(F), "rH": sum(cross[s] for s in H) / len(H)}


def concentration(seeds, topic, author, participants, topics):
    out = {}
    for t in topics:
        L = [s for s in seeds if topic[s] == t]
        if not L:
            continue
        c = np.array(sorted(Counter(author[s] for s in L).values(), reverse=True), float)
        p = c / c.sum()
        part = set().union(*[participants[s] for s in L])
        out[t] = {"posts": len(L), "authors": len(c), "N_eff": float(1 / (p ** 2).sum()),
                  "top1": float(p[:1].sum()), "top5": float(p[:5].sum()),
                  "cross_participants": len(part)}
    return out


def analyse(seeds, topic, author, feed, cross, participants):
    topics = sorted({topic[s] for s in seeds})
    o, d = stats(seeds, topic, author, feed, cross, topics)
    conc = concentration(seeds, topic, author, participants, topics)
    kit = kitagawa(o, "ransomware", rest_of(seeds, topic, feed, cross, "ransomware")) if "ransomware" in o else None
    # bootstrap over authors
    by = defaultdict(list)
    for s in seeds:
        by[author[s]].append(s)
    groups = list(by.values())
    bs = defaultdict(list)
    for _ in range(B):
        draws = rng.integers(0, len(groups), len(groups))
        samp = [s for g in draws for s in groups[g]]
        ob, db_ = stats(samp, topic, author, feed, cross, topics)
        bs["D_red"].append(1 - db_["D_std"] / db_["D_raw"] if db_["D_raw"] else np.nan)
        # each resampled copy of an author is a distinct producer (otherwise duplicates merge and N_eff drops)
        cb = {}
        for t in topics:
            cnt = Counter(j for j, g in enumerate(draws) for s in groups[g] if topic[s] == t)
            if cnt:
                pp = np.array(list(cnt.values()), float) / sum(cnt.values())
                cb[t] = {"N_eff": float(1 / (pp ** 2).sum())}
        for t in topics:
            if t in ob:
                bs[f"r:{t}"].append(ob[t]["r"])
                bs[f"rstd:{t}"].append(ob[t]["r_std"])
            if t in cb:
                bs[f"neff:{t}"].append(cb[t]["N_eff"])
        if kit:
            try:
                kb = kitagawa(ob, "ransomware", rest_of(samp, topic, feed, cross, "ransomware"))
                bs["kit_comp_share"].append(kb["composition"] / kb["gap"] if kb["gap"] else np.nan)
            except ZeroDivisionError:
                pass
    q = lambda k: [round(float(x), 4) for x in np.nanpercentile(bs[k], [2.5, 97.5])]
    rank = lambda k: {t: r for r, t in enumerate(sorted(o, key=lambda t: (-o[t][k], t)), 1)}
    rk = {k: rank(k) for k in ("r", "r_std", "rH")}
    prank = {t: r for r, t in enumerate(sorted(conc, key=lambda t: (-conc[t]["posts"], t)), 1)}
    nrank = {t: r for r, t in enumerate(sorted(conc, key=lambda t: (-conc[t]["N_eff"], t)), 1)}
    crank = {t: r for r, t in enumerate(sorted(o, key=lambda t: (-o[t]["r"] * o[t]["n"], t)), 1)}
    from scipy.stats import spearmanr
    tl = sorted(o)
    rho = lambda a, b: round(float(spearmanr([a[t] for t in tl], [b[t] for t in tl]).statistic), 3)
    res = {"n": len(seeds), "dispersion": {**{k: round(v, 5) for k, v in d.items()},
                                           "reduction_std": round(1 - d["D_std"] / d["D_raw"], 4),
                                           "reduction_std_ci": q("D_red"),
                                           "reduction_nonfeed": round(1 - d["D_nonfeed"] / d["D_raw"], 4)},
           "topics": {t: {**{k: round(v, 4) for k, v in o[t].items()},
                          "r_ci": q(f"r:{t}"), "r_std_ci": q(f"rstd:{t}"),
                          "rank_r": rk["r"][t], "rank_r_std": rk["r_std"][t], "rank_rH": rk["rH"][t],
                          **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in conc[t].items()},
                          "N_eff_ci": q(f"neff:{t}"),
                          "rank_posts": prank[t], "rank_N_eff": nrank[t], "rank_cross_edges": crank[t]}
                      for t in o},
           "rank_corr": {"posts_vs_Neff": rho(prank, nrank), "posts_vs_cross_edges": rho(prank, crank),
                         "Neff_vs_cross_edges": rho(nrank, crank), "r_vs_r_std": rho(rk["r"], rk["r_std"])}}
    if kit:
        res["ransomware_kitagawa"] = {**{k: round(v, 4) for k, v in kit.items()},
                                      "composition_share": round(kit["composition"] / kit["gap"], 4),
                                      "composition_share_ci": q("kit_comp_share")}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    args = ap.parse_args()
    m, runs, seeds, oc, edges, info, tc = S.desc.load()
    pops, partial, ps, info2, beh, vol = S.load()
    M = pops["main"]
    topic = {s: info[s]["topic"] for s in M}
    author = {s: info[s]["author_id"] for s in M}
    HF = human_feeds(args.audit)
    cross = {s: S.inter(ps, s, True) for s in M}
    participants = defaultdict(set)
    for e in edges:
        s = e["seed_post_id"]
        if s in topic and e["edge_type"] in ("reply", "quote") and e["source_actor_id"] != author[s]:
            participants[s].add(e["source_actor_id"])
    with db.connect() as c:
        txt = {r["post_id"]: r["text"] for r in c.execute("SELECT post_id, text FROM posts WHERE post_id = ANY(%s)", (M,))}
    SC = [s for s in M if strict_canada(txt[s])]
    out = {}
    for name, F in (("human", HF), ("behavioral_v1", beh), ("volume_only", vol)):
        feed = {s: author[s] in F for s in M}
        out[f"all|{name}"] = analyse(M, topic, author, feed, cross, participants)
        out[f"strict_canada|{name}"] = analyse(SC, topic, author, feed, cross, participants)
    p = Path("outputs/interaction_experiment/www_gonogo/producer.json")
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
