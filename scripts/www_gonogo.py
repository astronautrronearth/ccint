"""WWW 2027 short-paper go/no-go analyses on run-475 neighbourhood data (main population, observed seeds).

Run from the repo root:
    python scripts/www_gonogo.py   → outputs/interaction_experiment/www_gonogo/gonogo.json

Read-only, CPU only, ~7 s. Seeds: main population (llm_v2d), observed = thread and quote
endpoints success/partial, same outcome/edge selection as interaction_descriptive.load().
- "self" edges: reply/quote whose author is the seed author (self-threads), reported both ways.
- feed exclusions: behavioral_v1 (uses zero likes+reposts, so partly outcome-linked),
  volume_only (>=5 relevant posts on >=4 days, outcome-free), actor_v1 intel_feed_*.
- CIs: author-clustered bootstrap (resample seed authors with all their seeds).
- perm: topic labels shuffled across seeds; global stat sum n_k (r_k-g)^2/g,
  per-topic max-statistic FWER on |r_k-g|*sqrt(n_k). Reproduces WWW2027_SHORT_PAPER.md §3 FWER.
"""
from pathlib import Path
import importlib.util, json, sys
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location("desc", "scripts/interaction_descriptive.py")
desc = importlib.util.module_from_spec(spec); spec.loader.exec_module(desc)
from ccint import db
from ccint.analytics.broadcast import detect_broadcast_authors, SQL as BSQL
from ccint.interaction.collect import EP_QUOTES, EP_THREAD

m, runs, seeds, oc, edges, info, tc = desc.load()
OK = {"success", "partial"}
def observed(s):
    v = oc.get(s)
    return bool(v) and v[EP_THREAD]["outcome"] in OK and v[EP_QUOTES]["outcome"] in OK
main = [s for s, r in seeds.items() if r["in_main"] == "true" and observed(s)]

ps = defaultdict(lambda: Counter())
for e in edges:
    s = e["seed_post_id"]; self_ = e["source_actor_id"] == info[s]["author_id"]
    if e["edge_type"] == "reply":
        ps[s]["rep"] += 1; ps[s]["rep_ns"] += not self_
        if e["depth"] == 1: ps[s]["d1"] += 1; ps[s]["d1_ns"] += not self_
    elif e["edge_type"] == "quote":
        ps[s]["q"] += 1; ps[s]["q_ns"] += not self_

def inter(s, ns): return ps[s]["rep_ns"] + ps[s]["q_ns"] if ns else ps[s]["rep"] + ps[s]["q"]
def conv(s, ns):  # original definition: d1 reply or quote (d1 empty => d2 empty)
    return (ps[s]["d1_ns"] if ns else ps[s]["d1"]) > 0 or (ps[s]["q_ns"] if ns else ps[s]["q"]) > 0
topic = {s: info[s]["topic"] for s in main}; author = {s: info[s]["author_id"] for s in main}

with db.connect() as c:
    beh = detect_broadcast_authors(c, label_version="llm_v2d").author_ids
    vol = {r["author_id"] for r in c.execute(BSQL, {"label_version": "llm_v2d", "since": None, "until": None,
           "min_lag_hours": 24}) if r["n_posts"] >= 5 and r["n_days"] >= 4}
    kev = {r["post_id"]: r["k"] for r in c.execute("""
        SELECT pc.post_id, bool_or(coalesce(cr.known_exploited,false)) k
        FROM post_cves pc LEFT JOIN cve_records cr USING (cve_id)
        WHERE pc.post_id = ANY(%s) GROUP BY 1""", (main,))}
feedtype = {s: info[s]["actor_type"] in ("intel_feed_raw", "intel_feed_amplified") for s in main}
EXCL = {"none": lambda s: False,
        "behavioral_v1": lambda s: author[s] in beh,
        "volume_only(>=5 posts,>=4 days)": lambda s: author[s] in vol,
        "actor_v1_intel_feed": lambda s: feedtype[s]}
rng = np.random.default_rng(0)

def boot_csr(ids, ns, B=10000):
    by = defaultdict(list)
    for s in ids: by[author[s]].append(conv(s, ns))
    groups = list(by.values()); n = len(groups)
    num = np.array([sum(g) for g in groups], float); den = np.array([len(g) for g in groups], float)
    idx = rng.integers(0, n, (B, n))
    est = num[idx].sum(1) / den[idx].sum(1)
    return [round(float(x), 4) for x in np.percentile(est, [2.5, 97.5])]

def boot_ratio(ids, k, ns, B=10000):
    """interaction-per-post in topic k / rest, resampling authors (all their seeds)."""
    by = defaultdict(lambda: np.zeros(4))
    for s in ids:
        a = by[author[s]]; i = topic[s] == k
        a += [inter(s, ns) * i, i, inter(s, ns) * (not i), not i]
    M = np.array(list(by.values())); n = len(M)
    idx = rng.integers(0, n, (B, n)); S = M[idx].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (S[:, 0] / S[:, 1]) / (S[:, 2] / S[:, 3])
    pt = (M[:, 0].sum() / M[:, 1].sum()) / (M[:, 2].sum() / M[:, 3].sum())
    return round(float(pt), 3), [round(float(x), 3) for x in np.nanpercentile(r, [2.5, 97.5])]

def perm(ids, ns, B=20000):
    t = np.array([topic[s] for s in ids]); y = np.array([inter(s, ns) for s in ids], float)
    ks = sorted(set(t)); T = np.array([ks.index(x) for x in t]); n = np.bincount(T, minlength=len(ks))
    g = y.mean()
    def stats(TT):
        r = np.bincount(TT, weights=y, minlength=len(ks)) / n
        return (n * (r - g) ** 2).sum() / g, np.abs(r - g) * np.sqrt(n), r
    D0, Z0, r0 = stats(T); Dn = np.empty(B); Zmax = np.empty(B)
    for b in range(B):
        D, Z, _ = stats(rng.permutation(T)); Dn[b] = D; Zmax[b] = Z.max()
    return {"D_obs": round(float(D0), 1), "p_global": float((1 + (Dn >= D0).sum()) / (B + 1)),
            "per_topic": {k: {"n": int(n[i]), "rate": round(float(r0[i]), 3), "x_global": round(float(r0[i] / g), 2),
                              "p_fwer": round(float((1 + (Zmax >= Z0[i]).sum()) / (B + 1)), 4)} for i, k in enumerate(ks)}}

out = {"n_main_observed": len(main), "feed_sets": {"behavioral_v1": len(beh), "volume_only": len(vol)}}
for name, ex in EXCL.items():
    ids = [s for s in main if not ex(s)]
    for ns in (False, True):
        key = f"{name} | {'no self-replies/quotes' if ns else 'all edges'}"
        T = defaultdict(list)
        for s in ids: T[topic[s]].append(s)
        rows = {}
        for k, L in T.items():
            acts = {author[s] for s in L}
            rows[k] = {"posts": len(L), "authors": len(acts), "interactions": sum(inter(s, ns) for s in L),
                       "per_post": round(sum(inter(s, ns) for s in L) / len(L), 3),
                       "CSR": round(np.mean([conv(s, ns) for s in L]), 4), "CSR_ci_author": boot_csr(L, ns, 4000)}
        for col in ("posts", "authors", "interactions"):
            for r, k in enumerate(sorted(rows, key=lambda k: -rows[k][col]), 1): rows[k][f"rank_{col}"] = r
        tk = list(rows); v = lambda c: [rows[k][c] for k in tk]
        res = {"n": len(ids), "removed_seeds": len(main) - len(ids),
               "CSR_all": round(np.mean([conv(s, ns) for s in ids]), 4), "CSR_all_ci_author": boot_csr(ids, ns),
               "rho_post_inter": round(float(spearmanr(v("posts"), v("interactions")).statistic), 3),
               "rho_post_author": round(float(spearmanr(v("posts"), v("authors")).statistic), 3),
               "topics": dict(sorted(rows.items(), key=lambda kv: -kv[1]["posts"]))}
        if "ransomware" in rows:
            res["ransomware_vs_rest_ratio_ci_author"] = boot_ratio(ids, "ransomware", ns)
        if name in ("none", "behavioral_v1") or ns is False and name == "volume_only(>=5 posts,>=4 days)":
            res["perm"] = perm(ids, ns, 20000)
        out[key] = res
# ransomware feed composition
rw = [s for s in main if topic[s] == "ransomware"]
out["ransomware_composition"] = {"seeds": len(rw), "behavioral_v1": sum(author[s] in beh for s in rw),
    "volume_only": sum(author[s] in vol for s in rw), "actor_v1_intel_feed": sum(feedtype[s] for s in rw),
    "self_edges": sum(ps[s]["rep"] - ps[s]["rep_ns"] + ps[s]["q"] - ps[s]["q_ns"] for s in rw),
    "all_edges": sum(inter(s, False) for s in rw)}
out["self_edges_all_topics"] = {"self": sum(inter(s, False) - inter(s, True) for s in main),
                                "all": sum(inter(s, False) for s in main)}
# KEV anchor
grp = {"KEV_cve": [s for s in main if kev.get(s) is True], "nonKEV_cve": [s for s in main if kev.get(s) is False]}
out["kev"] = {g: {"posts": len(L), "authors": len({author[s] for s in L}),
                  "per_post": round(np.mean([inter(s, False) for s in L]), 3) if L else None,
                  "CSR": round(np.mean([conv(s, False) for s in L]), 4) if L else None,
                  "CSR_ci_author": boot_csr(L, False) if L else None,
                  "beh_feed_share": round(np.mean([author[s] in beh for s in L]), 3) if L else None}
              for g, L in grp.items()}
OUT = Path("outputs/interaction_experiment/www_gonogo"); OUT.mkdir(parents=True, exist_ok=True)
(OUT / "gonogo.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
print(f"wrote {OUT / 'gonogo.json'}")
