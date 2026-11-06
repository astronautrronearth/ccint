"""Replication at scale and across countries (PREREG: www2027/results/frozen_scale/PREREG.md).

    python scripts/www_scale.py --audit outputs/www_audit/scoring            # scale runs (purpose 'scale')
    python scripts/www_scale.py --audit ... --runs 475 --out /tmp/x.json     # dry run on the main run
        → outputs/interaction_experiment/www_gonogo/scale.json

Per stratum (all, ten country filters, no_country): feed share, Spearman(posts, accounts) and
Spearman(posts, N_eff) over topics, self share and continuation share, CSR, producer-standardised
dispersion reduction (author-clustered bootstrap, B = 1000), Kitagawa composition share of ransomware's gap,
and the cross-user rank breakdown point. Feeds: volume-only and behavioural rules recomputed on P_scale.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import zlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path[:0] = ["src", "scripts"]
from ccint import db  # noqa: E402
from ccint.interaction import manifest as mf  # noqa: E402
from ccint.interaction.collect import EP_QUOTES, EP_THREAD  # noqa: E402
from ccint.interaction.derive import DERIVATION_VERSION  # noqa: E402
from country_filters import COUNTRIES, countries  # noqa: E402


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _mod("prod", "scripts/www_producer.py")
E = _mod("exp", "scripts/www_experiments.py")
OK = {"success", "partial"}
B = 1000
rng = np.random.default_rng(20261106)
MIN_N = 30   # PREREG: strata < 300 seeds are reported (computed) but not interpreted


def scale_runs(sha):
    with db.connect() as c:
        return [r["run_id"] for r in c.execute(
            "SELECT run_id FROM neighborhood_runs WHERE purpose = 'scale' AND manifest_sha256 = %s ORDER BY run_id", (sha,))]


def load(runs, ids):
    with db.connect() as c:
        oc = defaultdict(dict)
        for r in c.execute("""SELECT collection_run_id, seed_post_id, endpoint, outcome FROM interaction_seed_outcomes
                              WHERE collection_run_id = ANY(%s) AND seed_post_id = ANY(%s) ORDER BY collection_run_id""",
                           (runs, ids)):
            prev = oc[r["seed_post_id"]].get(r["endpoint"])
            if prev is None or r["outcome"] != "failed":
                oc[r["seed_post_id"]][r["endpoint"]] = r
        use = {s: v[EP_THREAD]["collection_run_id"] for s, v in oc.items() if EP_THREAD in v}
        obs = [s for s, v in oc.items() if v.get(EP_THREAD, {}).get("outcome") in OK and v.get(EP_QUOTES, {}).get("outcome") in OK]
        edges = [e for e in c.execute("""SELECT collection_run_id, seed_post_id, edge_type, depth, source_post_uri,
                                                source_actor_id, target_post_uri, event_created_at
                                         FROM interaction_edges WHERE collection_run_id = ANY(%s) AND derivation_version = %s""",
                                      (runs, DERIVATION_VERSION)) if use.get(e["seed_post_id"]) == e["collection_run_id"]]
        info = {r["post_id"]: r for r in c.execute("""
            SELECT p.post_id, p.author_id, p.published_at, p.text, p.like_count, p.repost_count,
                   extract(epoch FROM (p.collected_at - p.published_at)) / 3600.0 AS lag_h,
                   coalesce(t.topic_key, CASE WHEN t.post_id IS NULL THEN 'no_topic' ELSE 'unassigned' END) AS topic,
                   coalesce(NULLIF(r.topic_key, 'other'), 'unassigned') AS rules_topic
            FROM posts p
            LEFT JOIN post_labels t ON t.post_id = p.post_id AND t.label_version = 'topic_v2s'
            LEFT JOIN post_labels r ON r.post_id = p.post_id AND r.label_version = 'rules_v2'
            WHERE p.post_id = ANY(%s)""", (ids,))}
    return obs, edges, info


def feed_rules(ids, info):
    """Volume-only and behavioural rules on the whole manifest population (broadcast.py thresholds)."""
    by = defaultdict(list)
    for s in ids:
        by[info[s]["author_id"]].append(info[s])
    vol, beh = set(), set()
    for a, L in by.items():
        if len(L) >= 5 and len({r["published_at"].date() for r in L}) >= 4:
            vol.add(a)
            st = [r for r in L if r["lag_h"] is not None and r["lag_h"] >= 24]
            if st and sum((r["like_count"] or 0) == 0 and (r["repost_count"] or 0) == 0 for r in st) / len(st) >= 0.6:
                beh.add(a)
    return {"volume_only": vol, "behavioral_v1": beh}


def dispersion_boot(seeds, topic, author, feed, cross):
    """Point estimate and author-clustered bootstrap of the standardisation reduction, vectorised."""
    topics = sorted({topic[s] for s in seeds})
    ti = {t: i for i, t in enumerate(topics)}
    auth = sorted({author[s] for s in seeds})
    ai = {a: i for i, a in enumerate(auth)}
    M = np.zeros((len(auth), 4, len(topics)))   # n, nF, cx, cxF per author x topic
    for s in seeds:
        a, t = ai[author[s]], ti[topic[s]]
        M[a, 0, t] += 1
        M[a, 2, t] += cross[s]
        if feed[s]:
            M[a, 1, t] += 1
            M[a, 3, t] += cross[s]

    def red(W):
        n, nF, cx, cxF = W
        nH, cxH = n - nF, cx - cxF
        ok = n > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            rF = np.where(nF > 0, cxF / np.maximum(nF, 1), np.nan)
            rH = np.where(nH > 0, cxH / np.maximum(nH, 1), np.nan)
        rF = np.where(np.isnan(rF), rH, rF)
        rH = np.where(np.isnan(rH), rF, rH)
        sbar = nF.sum() / n.sum()
        r = np.where(ok, cx / np.maximum(n, 1), 0)
        rs = sbar * rF + (1 - sbar) * rH
        w = n[ok]

        def d(x):
            x = x[ok]
            m = (w * x).sum() / w.sum()
            return (w * (x - m) ** 2).sum() / w.sum()
        dr = d(r)
        return 1 - d(rs) / dr if dr else np.nan

    point = red(M.sum(0))
    bs = [red(np.tensordot(rng.multinomial(len(auth), np.full(len(auth), 1 / len(auth))), M, axes=(0, 0))) for _ in range(B)]
    return round(float(point), 4), [round(float(x), 4) for x in np.nanpercentile(bs, [2.5, 97.5])]


def stratum(seeds, info, edges_by, feeds, cross, participants, topic_key, name=""):
    # per-stratum random streams, so a stratum's intervals do not depend on which strata ran before it
    global rng
    rng = np.random.default_rng([20261106, zlib.crc32(f"{name}|{topic_key}".encode())])
    E.rng = np.random.default_rng([20261107, zlib.crc32(f"{name}|{topic_key}".encode())])
    topic = {s: info[s][topic_key] for s in seeds}
    author = {s: info[s]["author_id"] for s in seeds}
    out = {"n": len(seeds), "accounts": len(set(author.values()))}
    if len(seeds) < 30:
        return out
    # concentration and rank correlations
    conc = P.concentration(seeds, topic, author, participants, sorted(set(topic.values())))
    tl = sorted(conc)
    out["rho_posts_accounts"] = round(float(spearmanr([conc[t]["posts"] for t in tl], [conc[t]["authors"] for t in tl]).statistic), 3)
    out["rho_posts_neff"] = round(float(spearmanr([conc[t]["posts"] for t in tl], [conc[t]["N_eff"] for t in tl]).statistic), 3)
    out["topics"] = {t: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in conc[t].items()} for t in tl}
    # self edges and their mechanism
    tot = sum(1 for s in seeds for e in edges_by[s])
    self_ = sum(1 for s in seeds for e in edges_by[s] if e["source_actor_id"] == author[s])
    mech = E.self_mechanism(seeds, info, [e for s in seeds for e in edges_by[s]]) if self_ else None
    out["edges"], out["self_edges"] = tot, self_
    out["self_share"] = round(self_ / tot, 4) if tot else None
    out["continuation_share"] = mech["shares"].get("continuation") if mech else None
    d1 = lambda s, ns: any(e["edge_type"] == "quote" or e["depth"] == 1 for e in edges_by[s]
                           if not ns or e["source_actor_id"] != author[s])
    out["CSR_raw"] = round(np.mean([d1(s, False) for s in seeds]), 4)
    out["CSR_nonself"] = round(np.mean([d1(s, True) for s in seeds]), 4)
    # feeds, composition, breakdown
    out["feeds"] = {}
    for name, F in feeds.items():
        feed = {s: author[s] in F for s in seeds}
        f = {"feed_share": round(np.mean(list(feed.values())), 4)}
        for t in ("ransomware", "vulnerability_cve"):
            L = [s for s in seeds if topic[s] == t]
            f[f"feed_share_{t}"] = round(np.mean([feed[s] for s in L]), 4) if L else None
        if len(seeds) >= MIN_N:
            f["std_reduction"], f["std_reduction_ci"] = dispersion_boot(seeds, topic, author, feed, cross)
            o, _ = P.stats(seeds, topic, author, feed, cross, sorted(set(topic.values())))
            if "ransomware" in o and sum(topic[s] != "ransomware" for s in seeds):
                try:
                    k = P.kitagawa(o, "ransomware", P.rest_of(seeds, topic, feed, cross, "ransomware"))
                    f["kitagawa_composition_share"] = round(k["composition"] / k["gap"], 4) if k["gap"] else None
                except ZeroDivisionError:
                    f["kitagawa_composition_share"] = None
        out["feeds"][name] = f
    if len(seeds) >= MIN_N:
        E.R = 200
        br = E.breakdown(seeds, topic, author, cross)
        ks = [v["cross_edges"]["k_star"] for v in br.values() if v.get("cross_edges")]
        out["breakdown_cross_le2"] = sum(k is not None and k <= 2 for k in ks)
        out["breakdown_cross_topics"] = len(ks)
        # topics that can fall two places (drop_needed == 2); the count above also includes near-bottom topics
        k2 = [v["cross_edges"]["k_star"] for v in br.values() if v.get("cross_edges") and v["cross_edges"]["drop_needed"] == 2]
        out["breakdown_cross_two_le2"] = sum(k is not None and k <= 2 for k in k2)
        out["breakdown_cross_two_topics"] = len(k2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--runs", type=int, nargs="*")
    ap.add_argument("--out", type=Path, default=Path("outputs/interaction_experiment/www_gonogo/scale.json"))
    a = ap.parse_args()
    m = mf.load("www_scale")
    ids = [int(r["post_id"]) for r in m.population("in_scale")]
    runs = a.runs or scale_runs(m.sha256)
    obs, edges, info = load(runs, ids)
    feeds = feed_rules(ids, info)
    seeds = sorted(obs)
    edges_by = defaultdict(list)
    for e in edges:
        if e["edge_type"] in ("reply", "quote") and e["seed_post_id"] in info:
            edges_by[e["seed_post_id"]].append(e)
    cross = {s: sum(e["source_actor_id"] != info[s]["author_id"] for e in edges_by[s]) for s in seeds}
    participants = {s: {e["source_actor_id"] for e in edges_by[s] if e["source_actor_id"] != info[s]["author_id"]} for s in seeds}
    strata = {"all": seeds}
    cs = {s: countries(info[s]["text"]) for s in seeds}
    for k in COUNTRIES:
        strata[k] = [s for s in seeds if k in cs[s]]
    strata["no_country"] = [s for s in seeds if not cs[s]]
    # human labels: rule precision/recall on audited accounts present here
    HF = P.human_feeds(a.audit)
    key = {r["author_id"] for r in csv.DictReader(open(a.audit / "feed/key.csv"))}
    present = key & {info[s]["author_id"] for s in seeds}
    rules_vs_human = {}
    for name, F in feeds.items():
        tp = len(present & F & HF)
        rules_vs_human[name] = {"audited_accounts_present": len(present), "precision": round(tp / len(present & F), 4) if present & F else None,
                                "recall": round(tp / len(present & HF), 4) if present & HF else None}
    out = {"runs": runs, "manifest_sha256": m.sha256, "n_manifest": len(ids), "n_observed": len(seeds),
           "feeds_accounts": {k: len(v) for k, v in feeds.items()}, "rules_vs_human": rules_vs_human,
           "strata": {}, "strata_rules_topic": {}}
    for k, L in strata.items():
        out["strata"][k] = stratum(L, info, edges_by, feeds, cross, participants, "topic", name=k)
        print(k, len(L), flush=True)
    out["strata_rules_topic"]["all"] = stratum(seeds, info, edges_by, feeds, cross, participants, "rules_topic", name="all")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=str))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
