"""Four strengthening experiments for the WWW short paper, as pre-registered in
www2027/results/frozen_1107/PREREG.md (2026-11-06).

    python scripts/www_experiments.py --audit outputs/www_audit/scoring
        → outputs/interaction_experiment/www_gonogo/experiments.json

E1 burst audit, E2 rank breakdown point, E3 volume-matched placebo, E4 self-thread mechanism.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path[:0] = ["src", "scripts"]


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _mod("sens", "scripts/www_sensitivity.py")
P = _mod("prod", "scripts/www_producer.py")

R = 1000
rng = np.random.default_rng(20261106)
BINS = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 10 ** 9)]


def rank_of(counts, topics):
    """Competition ranks: ties share the better rank (matches the paper's Table 1)."""
    return {t: 1 + sum(counts.get(u, 0) > counts.get(t, 0) for u in topics) for t in topics}


# ---------------------------------------------------------------- E1
def bursts(seeds, topic, author, feed, participants, pub, k):
    days = [date(2026, 8, 22) + timedelta(d) for d in range((date(2026, 9, 23) - date(2026, 8, 22)).days + 1)]
    out = {"bursts": 0, "survive_nonfeed": 0, "survive_participants": 0, "neff_burst": [], "neff_other": [],
           "tested_days": 0, "participant_bursts_any": 0, "nonfeed_bursts_any": 0,
           "feed_share_burst": [], "by_topic": Counter()}
    for t in sorted({topic[s] for s in seeds}):
        L = [s for s in seeds if topic[s] == t]
        byday = defaultdict(list)
        for s in L:
            byday[pub[s]].append(s)
        a = np.array([len(byday[d]) for d in days], float)
        h = np.array([sum(not feed[s] for s in byday[d]) for d in days], float)
        c = np.array([len(set().union(*[participants[s] for s in byday[d]])) if byday[d] else 0 for d in days], float)

        def is_burst(x, i, floor_min):
            base = x[i - 7:i]
            med = np.median(base)
            mad = max(np.median(np.abs(base - med)), 1.0)
            return x[i] > med + k * mad and x[i] >= floor_min

        for i in range(7, len(days)):
            L_d = byday[days[i]]
            cnt = np.array(list(Counter(author[s] for s in L_d).values()), float)
            neff = float(1 / ((cnt / cnt.sum()) ** 2).sum()) if len(cnt) else None
            out["tested_days"] += 1
            out["participant_bursts_any"] += bool(is_burst(c, i, 1))
            out["nonfeed_bursts_any"] += bool(is_burst(h, i, 1))
            if is_burst(a, i, 5):
                out["bursts"] += 1
                out["by_topic"][t] += 1
                out["survive_nonfeed"] += bool(is_burst(h, i, 1))
                out["survive_participants"] += bool(is_burst(c, i, 1))
                out["neff_burst"].append(neff)
                out["feed_share_burst"].append(float(np.mean([feed[s] for s in L_d])))
            elif neff is not None:
                out["neff_other"].append(neff)
    b = out["bursts"]
    return {"k": k, "bursts": b, "by_topic": dict(out["by_topic"]),
            "share_survive_nonfeed": round(out["survive_nonfeed"] / b, 4) if b else None,
            "share_survive_participants": round(out["survive_participants"] / b, 4) if b else None,
            "median_neff_burst_day": round(float(np.median(out["neff_burst"])), 2) if b else None,
            "median_neff_other_days": round(float(np.median(out["neff_other"])), 2),
            # post hoc (2026-11-06): base rates for the coincidence shares above
            "tested_topic_days": out["tested_days"],
            "base_rate_participant_burst": round(out["participant_bursts_any"] / out["tested_days"], 4),
            "base_rate_nonfeed_burst": round(out["nonfeed_bursts_any"] / out["tested_days"], 4),
            "median_feed_share_burst_day": round(float(np.median(out["feed_share_burst"])), 3) if b else None}


# ---------------------------------------------------------------- E2
def breakdown(seeds, topic, author, cross):
    topics = sorted({topic[s] for s in seeds})
    posts = Counter(topic[s] for s in seeds)
    cedges = Counter()
    for s in seeds:
        cedges[topic[s]] += cross[s]
    res = {}
    for unit, base in (("posts", posts), ("cross_edges", cedges)):
        r0 = rank_of(base, topics)
        for t in topics:
            if r0[t] == len(topics):
                res.setdefault(t, {})[unit] = None
                continue
            need = 2 if r0[t] <= len(topics) - 2 else 1
            L = [s for s in seeds if topic[s] == t]
            contrib = Counter()
            for s in L:
                contrib[author[s]] += 1 if unit == "posts" else cross[s]
            order = [a for a, _ in sorted(contrib.items(), key=lambda x: (-x[1], x[0]))]
            cur = dict(base)
            removed_posts = 0
            kstar = None
            for i, a in enumerate(order, 1):
                mine = [s for s in L if author[s] == a]
                removed_posts += len(mine)
                cur[t] -= len(mine) if unit == "posts" else sum(cross[s] for s in mine)
                if rank_of(cur, topics)[t] - r0[t] >= need:
                    kstar = i
                    break
            # control: remove the same number of posts at random from t
            hit = 0
            if kstar is not None:
                arr = np.array([1 if unit == "posts" else cross[s] for s in L], float)
                for _ in range(R):
                    idx = rng.choice(len(L), removed_posts, replace=False)
                    cur2 = dict(base)
                    cur2[t] -= arr[idx].sum()
                    hit += rank_of(cur2, topics)[t] - r0[t] >= need
            res.setdefault(t, {})[unit] = {"rank": r0[t], "drop_needed": need, "k_star": kstar,
                                           "authors": len(order), "posts_removed": removed_posts if kstar else None,
                                           "random_same_posts_p": round(hit / R, 4) if kstar else None}
    return res


# ---------------------------------------------------------------- E3
def removal_stats(seeds, topic, cross, keep):
    topics = sorted({topic[s] for s in seeds})
    K = [s for s in seeds if keep(s)]
    p0 = rank_of(Counter(topic[s] for s in seeds), topics)
    p1 = rank_of(Counter(topic[s] for s in K), topics)
    rho = float(spearmanr([p0[t] for t in topics], [p1[t] for t in topics]).statistic)

    def disp(L):
        n = Counter(topic[s] for s in L)
        cx = Counter()
        for s in L:
            cx[topic[s]] += cross[s]
        ts = [t for t in topics if n[t]]
        w = np.array([n[t] for t in ts], float)
        r = np.array([cx[t] / n[t] for t in ts])
        m = (w * r).sum() / w.sum()
        return float((w * (r - m) ** 2).sum() / w.sum()), cx, n

    d0, _, _ = disp(seeds)
    d1, cx, n = disp(K)
    rw = (cx["ransomware"] / n["ransomware"]) / ((sum(cx.values()) - cx["ransomware"]) / (sum(n.values()) - n["ransomware"])) \
        if n["ransomware"] else None
    return {"rho_posts_rank": rho, "rw_rank_posts": p1["ransomware"], "disp_ratio": d1 / d0, "rw_ratio": rw,
            "removed": len(seeds) - len(K)}


def placebo(seeds, topic, author, cross, F):
    npost = Counter(author[s] for s in seeds)
    feeds = [a for a in npost if a in F]
    pool = defaultdict(list)
    for a in npost:
        if a not in F:
            pool[next(i for i, (lo, hi) in enumerate(BINS) if lo <= npost[a] <= hi)].append(a)
    real = removal_stats(seeds, topic, cross, lambda s: author[s] not in F)
    draws = defaultdict(list)
    matched = []
    for _ in range(R):
        avail = {b: list(v) for b, v in pool.items()}
        X = set()
        for a in feeds:
            b = next(i for i, (lo, hi) in enumerate(BINS) if lo <= npost[a] <= hi)
            while b >= 0 and not avail.get(b):
                b -= 1
            if b < 0:
                continue
            j = rng.integers(len(avail[b]))
            X.add(avail[b].pop(j))
        st = removal_stats(seeds, topic, cross, lambda s: author[s] not in X)
        matched.append(st["removed"])
        for k, v in st.items():
            draws[k].append(v)
    # post hoc (2026-11-06): how many accounts per stratum, to show why volume matching falls short
    strata = {f"{lo}-{hi if hi < 10**9 else 'max'}": {"feeds": sum(lo <= npost[a] <= hi for a in feeds),
              "nonfeeds": sum(lo <= npost[a] <= hi for a in npost if a not in F),
              "feed_posts": sum(npost[a] for a in feeds if lo <= npost[a] <= hi)} for lo, hi in BINS}
    out = {"strata": strata, "real": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in real.items()},
           "placebo_removed_posts_median": int(np.median(matched)), "stats": {}}
    for k in ("rho_posts_rank", "rw_rank_posts", "disp_ratio", "rw_ratio"):
        x = np.array([v for v in draws[k] if v is not None], float)
        out["stats"][k] = {"real": round(float(real[k]), 4), "placebo_median": round(float(np.median(x)), 4),
                           "placebo_95": [round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])],
                           "real_percentile": round(float((x < real[k]).mean() + 0.5 * (x == real[k]).mean()), 4)}
    return out


# ---------------------------------------------------------------- E4
def self_mechanism(seeds, info, edges):
    S_ = set(seeds)
    d1 = {}
    for e in edges:
        if e["seed_post_id"] in S_ and e["edge_type"] == "reply" and e["depth"] == 1:
            d1[(e["seed_post_id"], e["source_post_uri"])] = e["source_actor_id"]
    cls = Counter()
    delay = defaultdict(list)
    resp_seeds = set()
    for e in edges:
        s = e["seed_post_id"]
        if s not in S_ or e["edge_type"] not in ("reply", "quote"):
            continue
        a = info[s]["author_id"]
        if e["source_actor_id"] != a:
            continue
        if e["edge_type"] == "quote":
            c = "self_quote"
        elif e["depth"] == 1:
            c = "continuation"
        else:
            parent = d1.get((s, e["target_post_uri"]))
            c = "continuation" if parent == a else ("response" if parent is not None else "unknown_parent")
            if c == "response":
                resp_seeds.add(s)
        cls[c] += 1
        if e["event_created_at"] and info[s]["published_at"]:
            delay[c].append((e["event_created_at"] - info[s]["published_at"]).total_seconds() / 60)
    tot = sum(cls.values())
    out = {"self_edges": tot, "classes": dict(cls),
           "shares": {c: round(v / tot, 4) for c, v in cls.items()}}
    alld = [x for v in delay.values() for x in v]
    out["delay_minutes"] = {c: {"median": round(float(np.median(v)), 1), "share_le_10min": round(float(np.mean(np.array(v) <= 10)), 4)}
                            for c, v in delay.items()}
    out["delay_minutes"]["all"] = {"median": round(float(np.median(alld)), 1),
                                   "share_le_10min": round(float(np.mean(np.array(alld) <= 10)), 4)}
    out["seeds_with_self_response"] = len(resp_seeds)
    out["delays_minutes"] = {c: [round(x, 3) for x in v] for c, v in delay.items()}  # for Figure 2
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    args = ap.parse_args()
    m, runs, seeds_all, oc, edges, info, tc = S.desc.load()
    pops, partial, ps, info2, beh, vol = S.load()
    M = pops["main"]
    topic = {s: info[s]["topic"] for s in M}
    author = {s: info[s]["author_id"] for s in M}
    pub = {s: info[s]["published_at"].date() for s in M}
    cross = {s: S.inter(ps, s, True) for s in M}
    participants = defaultdict(set)
    for e in edges:
        s = e["seed_post_id"]
        if s in topic and e["edge_type"] in ("reply", "quote") and e["source_actor_id"] != author[s]:
            participants[s].add(e["source_actor_id"])
    HF = P.human_feeds(args.audit)
    defs = {"human": HF, "behavioral_v1": beh, "volume_only": vol}
    out = {"n": len(M), "E1": {}, "E3": {}}
    for name, F in defs.items():
        feed = {s: author[s] in F for s in M}
        out["E1"][name] = [bursts(M, topic, author, feed, participants, pub, k) for k in (2.5, 3.0, 3.5)]
        out["E3"][name] = placebo(M, topic, author, cross, F)
        print("done", name, flush=True)
    out["E2"] = breakdown(M, topic, author, cross)
    out["E4"] = self_mechanism(M, info, edges)
    p = Path("outputs/interaction_experiment/www_gonogo/experiments.json")
    p.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
