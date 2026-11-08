"""Benchmark attack evaluation of the frozen learned methods (PREREG_BENCH.md, 2026-11-08).
Generated from scripts/www_manip_adaptive.py (attacks, victims, seeds, draws unchanged) plus two methods:
ranker (scripts/www_ranker.py, frozen weights trained on the five development settings) and gate
(scripts/www_gate.py, trained on development days; leave-one-setting-out for development settings).
Both read per-post reply counts; placement of fake replies on the victim topic:
  --place top     (primary) all fake replies and staged answers on the victim's most-replied real post
  --place spread  (secondary) spread as evenly as possible over the victim's real posts
Fake posts (A10-A13) are added with zero replies. hist_y is not attacked.
Options --victim low|near as in www_manip_adaptive.py. Output bench_attack{_spread}{_near}.json.
Original header of the attack script follows.
Options: --victim low (default: a lower-half topic by true yield) | near (true-yield rank 4-6, just outside the
best pick); --poison f (a random share f of history topic-days carries 450 extra unanswered replies when the caps
are calibrated, i.e. a campaign that ran during the history window). Success = share of attacked days on which the
victim enters the count's top 3.
EXPLORATORY COPY (2026-11-07, after RESULTS_DOSE3): adaptive all-sides attack A12 and per-day yields.
A13 = A12 with staged answers: B fresh repliers each answered once by a second fresh account (+B replies),
so every fake replier is in C_recv; the B fresh accounts also post once. A12 = A9a + A10 on the same victim: B fresh accounts each post once (+A, +V) and reply R/B times to unanswered real
depth-1 replies (+R, +C_recv, +C). Budgets B30/R45, B100/R150, B300/R450. Output manipulation_adaptive.json.
Synthetic engagement manipulation (stress test for listener-side counts), paper 2.

Motivation: reply counts are the near-optimal signal on clean data but are the easiest to inflate (coordinated reply
accounts). The producer dial applied to the listener side gives a robust count: the effective number of responders
D_q = (sum_j p_j^q)^(1/(1-q)) over the accounts j replying to a unit's posts before the cutoff, p_j = j's share of
those replies (q = 2: inverse Simpson; q -> 1: exp Shannon; q = 0: distinct responders).

Attack: on each evaluation day, one unit among those whose true next-day yield is in the lower half is chosen at
random and receives R fake reply edges from B fake accounts (evenly split), created before the cutoff. Fake accounts
are never counted as genuine new responders, so the target is unchanged. Grid: B in {1, 3, 10}, R in {0, 50, 200,
1000}; 20 random draws per cell.
Rankings: edges (all replies+quotes), distinct responders, effective responders D_1 and D_2, authors, posts, oracle.

    python scripts/www_manipulation.py [--known-only] → outputs/interaction_experiment/www_eval/manipulation{_known}.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("mech", "scripts/www_dose2_mech.py")
MECH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MECH)
MON = MECH.MON
spec_rk = importlib.util.spec_from_file_location("rk", "scripts/www_ranker.py")
RK = importlib.util.module_from_spec(spec_rk)
spec_rk.loader.exec_module(RK)
BF = RK.BF
PLACE = "top"
W_RK = None          # frozen ranker weights (set in main)
GATE = {}            # setting -> gate predictor (set in main)
CUR = None           # current setting name
SIGN = ["posts", "authors", "edges", "habitual", "peak5"]   # order of www_gate.SIG


def gate_sig(V, A, E, hy, e):
    return [V, A, E, hy, np.sort(e)[::-1][:5].sum()]


def gate_dayfeat(M, allE):
    """www_gate.dayfeat on a matrix of signals (rows topics, columns SIGN) and post-level engagement."""
    E = M[:, 2]
    ag = []
    for i in range(M.shape[1]):
        for j in range(i + 1, M.shape[1]):
            r = spearmanr(M[:, i], M[:, j]).correlation
            ag.append(0.0 if np.isnan(r) else r)
    return np.array([E.max() / max(1, E.sum()), allE.max() / max(1, allE.sum()), np.log(len(M)),
                     np.log1p(M[:, 0].sum()), np.log1p(E.sum()), *ag, 1.0])


def gate_fit(rows_by_setting, train):
    X, Y = [], []
    for n in train:
        for x, y in rows_by_setting[n]:
            X.append(x); Y.append(y / max(1, y.max()))
    X, Y = np.array(X), np.array(Y)
    mu, sd = X.mean(0), X.std(0) + 1e-6; sd[-1] = 1; mu[-1] = 0
    Xs = (X - mu) / sd
    W = np.linalg.solve(Xs.T @ Xs + 5 * np.eye(X.shape[1]), Xs.T @ Y)
    return lambda x: ((x - mu) / sd) @ W


def attacked_e(c, add):
    e = np.array(c["e"], float)
    if add <= 0 or len(e) == 0:
        return e
    e = e.copy()
    if PLACE == "top":
        e[int(np.argmax(e))] += add
    else:
        q, r = divmod(int(add), len(e))
        e += q
        order = np.argsort(-e - 1e-9 * np.arange(len(e)))
        e[order[:r]] += 1
    return e


N_SMALL = Counter()


def learned_scores(cells, T, sc, victim, add, nfake_posts):
    """Add sc[t]['ranker'] and sc[t]['gate'] for the day's topics under the attack (cells hold attacked A, V)."""
    es, Vs, As, hys = {}, {}, {}, {}
    for t in T:
        c = cells[(t, T_DAY)]
        e = attacked_e(c, add) if t == victim else np.array(c["e"], float)
        if t == victim and nfake_posts:
            e = np.concatenate([e, np.zeros(nfake_posts)])
        es[t], Vs[t], As[t], hys[t] = e, c["V"], c["A"], c["hist_y"]
    # ranker: within-day standardisation over the day's topics (www_ranker.standardise), score Z @ w
    F = np.array([[np.log1p(es[t].sum()), np.log1p(np.sort(es[t])[::-1][:5].sum()), np.log1p(hys[t])] for t in T])
    Z = (F - F.mean(0)) / (F.std(0) + 1e-6)
    for t, z in zip(T, Z):
        sc[t]["ranker"] = float(z @ W_RK)
    M = np.array([gate_sig(Vs[t], As[t], es[t].sum(), hys[t], es[t]) for t in T], float)
    if len(T) >= 4:
        allE = np.concatenate([es[t] for t in T])
        j = int(np.argmax(GATE[CUR](gate_dayfeat(M, allE))))
    else:
        N_SMALL[CUR] += 1
        j = 2                                                     # fewer than 4 topics: fall back to edges
    for t, m in zip(T, M):
        sc[t]["gate"] = float(m[j])
        sc[t]["_gate_sig"] = SIGN[j]


T_DAY = None
ACT = Counter()
KQ = {}
KR = {}                               # adaptive kappa per quantile, set per setting from history days only
QS = (0.5, 0.75, 0.9)
GRID_B = [1, 3, 10, 100, 300]        # 100, 300: Sybil attacks (many new accounts, few replies each)
GRID_R = [0, 50, 200, 1000]
DRAWS = 20


def cells_of(S):
    by = {}
    for s in S["seeds"]:
        r = S["info"][s]
        by.setdefault((r["topic"], r["published_at"].date()), []).append(s)
    cells = {}
    for (t, d), R in by.items():
        c = MON.cut(d)
        c2 = c + timedelta(days=1)
        before, after, resp = set(), set(), Counter()
        ne = 0
        conv = set()
        recv_by, d1u = {}, set()
        for s in R:
            a = S["info"][s]["author_id"]
            pre = [e for e in S["eb"][s] if e["event_created_at"] is not None and e["event_created_at"] < c]
            replied_to = {e["target_post_uri"] for e in pre if e.get("depth") == 2}
            for e in pre:
                if e["source_actor_id"] == a:
                    continue
                # a responder is in conversation if they replied to a reply, or their reply drew a reply
                if e.get("depth") == 2 or (e.get("depth") == 1 and e["source_post_uri"] in replied_to):
                    conv.add(e["source_actor_id"])
            # receipt-based (2026-11-07, after cold review W1): a depth-1 replier counts only if another account
            # answered their reply; answerers are kept to test whether they are established
            d1_author = {e["source_post_uri"]: e["source_actor_id"] for e in pre if e.get("depth") == 1}
            for e in pre:
                if e.get("depth") == 2 and e["target_post_uri"] in d1_author:
                    j = d1_author[e["target_post_uri"]]
                    if j != a and e["source_actor_id"] != j:
                        recv_by.setdefault(j, set()).add(e["source_actor_id"])
                        d1u.discard(j)
            for e in pre:
                if e.get("depth") == 1 and e["source_actor_id"] != a and e["source_actor_id"] not in recv_by:
                    d1u.add(e["source_actor_id"])
            for e in S["eb"][s]:
                ts = e["event_created_at"]
                if ts is None:
                    continue
                ext = e["source_actor_id"] != a
                if ts < c:
                    ne += 1
                    if ext:
                        before.add(e["source_actor_id"])
                        resp[e["source_actor_id"]] += 1
                elif ts < c2 and ext:
                    after.add(e["source_actor_id"])
        cells[(t, d)] = {"V": len(R), "A": len({S["info"][s]["author_id"] for s in R}), "E": ne, "resp": resp,
                         "T": after - before, "conv": conv,
                         "recv_by": recv_by, "d1u": d1u - set(recv_by)}
    # established responders: accounts that replied to any unit of the setting on an earlier day (before d's start)
    first_seen = {}
    for (t, d), c in sorted(cells.items(), key=lambda kv: kv[0][1]):
        for j in c["resp"]:
            first_seen.setdefault(j, d)
    seen_unit = {}
    for (t, d), c in sorted(cells.items(), key=lambda kv: kv[0][1]):
        prev = seen_unit.setdefault(t, set())
        c["prev_unit"] = set(prev)
        prev |= set(c["resp"])
    for (t, d), c in cells.items():
        c["recv"] = set(c["recv_by"])
        c["recv_est"] = {j for j, ks in c["recv_by"].items() if any(first_seen.get(k, d) < d for k in ks)}
        c["est"] = Counter({j: k for j, k in c["resp"].items() if first_seen.get(j, d) < d})
    # pool of established accounts per day (for sleeper attacks)
    pool = {}
    for (t, d), c in cells.items():
        pool.setdefault(d, set()).update(j for j in first_seen if first_seen[j] < d)
    for (t, d), c in cells.items():
        c["pool"] = pool[d]
    return cells


def hill(counts, q):
    v = np.array([x for x in counts if x > 0], float)
    if not len(v):
        return 0.0
    p = v / v.sum()
    if q == 0:
        return float(len(v))
    if q == 1:
        return float(np.exp(-(p * np.log(p)).sum()))
    return float((p ** q).sum() ** (1 / (1 - q)))


def conv_w(c, fake, fake_ids, w):
    real_new = [j for j in c["resp"] if j not in c["prev_unit"]]
    fk = [j for j in fake_ids if j not in c["prev_unit"]]
    conv_n = sum(j in c["conv"] for j in real_new) + sum(j in c["conv"] for j in fk)
    other_n = sum(j not in c["conv"] for j in real_new) + sum(j not in c["conv"] for j in fk) + len(fake)
    return float(conv_n + w * other_n)


def scores(c, fake, fake_est=None, fake_ids=None):
    """fake: reply counts of fake accounts that are new; fake_est: counts of fake accounts that are established
    (sleepers/pods); fake_ids: their ids (for new-to-unit filtering)."""
    fake_est = fake_est or []
    fake_ids = fake_ids or []
    counts = list(c["resp"].values()) + fake + fake_est
    est = list(c["est"].values()) + fake_est
    newu = [k for j, k in c["resp"].items() if j not in c["prev_unit"]] + fake + \
           [k for j, k in zip(fake_ids, fake_est) if j not in c["prev_unit"]]
    newu_est = [k for j, k in c["est"].items() if j not in c["prev_unit"]] + \
               [k for j, k in zip(fake_ids, fake_est) if j not in c["prev_unit"]]
    return {"edges": c["E"] + sum(fake) + sum(fake_est), "responders": len(c["resp"]) + len(fake) + len(fake_est),
            "eff_resp_D1": hill(counts, 1), "eff_resp_D2": hill(counts, 2),
            "est_resp": float(len(est)), "est_eff_D1": hill(est, 1),
            "new_unit_D1": hill(newu, 1), "new_unit_est_D1": hill(newu_est, 1),
            # conversational responders: fake replies are depth-1 broadcasts that draw no reply, so they never count
            "conv_resp": float(len(c["conv"] - set(fake_ids)) + len(set(fake_ids) & c["conv"])),
            # conversation-verified count: conversational new-to-unit responders weigh 1, other new-to-unit responders w
            "conv_w25": conv_w(c, fake, fake_ids, 0.25), "conv_w50": conv_w(c, fake, fake_ids, 0.5),
            "conv_new_unit": float(len([j for j in c["conv"] if j not in c["prev_unit"]])),
            "authors": c["A"], "posts": c["V"], "oracle": len(c["T"]),
            # conversation-anchored clipping (exploratory 2026-11-07 ~08:15): reply count capped at kappa x (1 + number
            # of conversational responders); a non-conversing attacker cannot raise the cap
            **{f"clip_k{k}": float(min(c["E"] + sum(fake) + sum(fake_est),
                                        k * (1 + len(c["conv"] - set(fake_ids)) + len(set(fake_ids) & c["conv"]))))
               for k in (2, 4, 8, 16)},
            # adaptive clipping (exploratory, after amendment 15's held-out reading): kappa = the setting's own
            # quantile q of R/(1+C) over (topic, day) cells on history days
            **{f"clipA_q{int(q*100)}": float(min(c["E"] + sum(fake) + sum(fake_est),
                                                 KQ[q] * (1 + len(c["conv"] - set(fake_ids)) + len(set(fake_ids) & c["conv"]))))
               for q in QS},
            # receipt-anchored (amendment 1 to PREREG_DOSE3): C = repliers answered by another account (recv) or by an
            # established account (recv_est); kappa = history q90 of R/(1+C) for the same C
            "anchR_q90": float(min(c["E"] + sum(fake) + sum(fake_est), KR["recv"] * (1 + len(c["recv"])))),
            "anchRE_q90": float(min(c["E"] + sum(fake) + sum(fake_est), KR["recv_est"] * (1 + len(c["recv_est"])))),
            "AxRE_b75": float(c["A"] ** 0.25 * (1 + min(c["E"] + sum(fake) + sum(fake_est),
                                                        KR["recv_est"] * (1 + len(c["recv_est"])))) ** 0.75),
            # ablations demanded by the cold review: anchor to authors; per-responder cap (trimmed replies)
            "anchA_q90": float(min(c["E"] + sum(fake) + sum(fake_est), KR["A"] * (1 + c["A"]))),
            **{f"anchA_q{q}": float(min(c["E"] + sum(fake) + sum(fake_est), KR[f"A{q}"] * (1 + c["A"]))) for q in (95, 99)},
            "anchAR_q90": float(min(c["E"] + sum(fake) + sum(fake_est), KR["A"] * (1 + c["A"]), KR["recv"] * (1 + len(c["recv"])))),
            "anchAR_q99": float(min(c["E"] + sum(fake) + sum(fake_est), KR["A99"] * (1 + c["A"]), KR["recv"] * (1 + len(c["recv"])))),
            "trim2": float(sum(min(k, 2) for k in c["resp"].values()) + sum(min(k, 2) for k in fake) +
                           sum(min(k, 2) for k in fake_est)),
            "trim1": float(len(c["resp"]) + len(fake) + len(fake_est)),
            # reach x propensity (exploratory): authors^(1-b) * clipped replies^b, clipped at the q90 adaptive kappa
            **{f"AxR_b{int(b*100)}": float(c["A"] ** (1 - b) * (1 + min(c["E"] + sum(fake) + sum(fake_est),
                   KQ[0.9] * (1 + len(c["conv"] - set(fake_ids)) + len(set(fake_ids) & c["conv"])))) ** b)
               for b in (0.25, 0.5, 0.75)}}


ATTACKS = {  # name: (kind, B accounts, R total fake replies); parameters from docs/revision/LITREV_1107/manipulation.md
    "clean": ("none", 0, 0),
    # A1 reply bombing (Pote et al. ICWSM 2025): few young accounts, tens to hundreds of replies each
    "A1_reply_bomb_B5_R200": ("new", 5, 200), "A1_reply_bomb_B5_R1000": ("new", 5, 1000),
    # A2 fresh Sybil swarm (Elmas et al. EuroS&P 2021; De Cristofaro et al. IMC 2014): many new accounts, 1-2 each
    "A2_sybil_B300_R450": ("new", 300, 450), "A2_sybil_B1000_R1500": ("new", 1000, 1500),
    # A3 aged sleepers (Thomas et al. USENIX Security 2013): purchased accounts with little prior activity
    "A3_sleeper_B100_R150": ("sleeper_low", 100, 150), "A3_sleeper_B500_R750": ("sleeper_low", 500, 750),
    # A4 compromised accounts (Elmas 2021): organic accounts with full history, 1 reply each
    "A4_compromised_B100_R100": ("sleeper_high", 100, 100), "A4_compromised_B500_R500": ("sleeper_high", 500, 500),
    # A5 engagement pod (Weerasinghe et al. WWW 2020): established accounts, same unit every day, 1 reply each
    "A5_pod_G50_R50": ("pod", 50, 50), "A5_pod_G350_R350": ("pod", 350, 350),
    # A6 sockpuppet pair (Kumar et al. WWW 2017): 2-3 accounts, a few replies
    "A6_sockpuppet_B3_R15": ("new", 3, 15),
    # A7 mobilised brigade (Kumar et al. WWW 2018): real low-activity accounts, 1-3 replies each
    "A7_brigade_B200_R400": ("sleeper_low", 200, 400),
    # A8 adaptive: established accounts that also reply to each other (fake conversation); breaks conversation checks
    "A8_fake_conversation_B100_R200": ("sleeper_conv", 100, 200),
    # A9 (cold review W1): accounts reply to real replies on the victim topic, so they "reply to a reply" and the real
    # repliers they answer now "drew a reply". A9a fresh accounts, A9b aged accounts (established, high activity)
    "A9a_reply_to_replies_B300_R450": ("r2r_new", 300, 450),
    "A9b_reply_to_replies_aged_B300_R450": ("r2r_aged", 300, 450),
    # A10 (cold review W3): fresh accounts post on the victim topic instead of replying (astroturf posting);
    # A11: the same accounts both post and reply-bomb (both sides inflated)
    "A10_post_flood_B300": ("post_flood", 300, 0),
    "A11_post_and_reply_B300_R450": ("post_reply", 300, 450),
}
ATTACKS.update({"A12_all_sides_B30_R45": ("all_sides", 30, 45), "A12_all_sides_B100_R150": ("all_sides", 100, 150),
    "A12_all_sides_B300_R450": ("all_sides", 300, 450),
    "A13_staged_all_B30_R45": ("all_staged", 30, 45), "A13_staged_all_B100_R150": ("all_staged", 100, 150),
    "A13_staged_all_B300_R450": ("all_staged", 300, 450)})


VICTIM, POISON = "low", 0.0


def run(S):
    global ACT
    cells = cells_of(S)
    cb = BF.prepare(S, {}, {})
    for k, c in cells.items():
        assert abs(cb[k]["E"] - c["E"]) < 1e-9, (k, cb[k]["E"], c["E"])
        c["e"], c["hist_y"] = cb[k]["e"], cb[k]["hist_y"]
    hd = set(S["hist"])
    prng = np.random.default_rng(20261109)
    keys_h = sorted(k for k in cells if k[1] in hd)
    pois = set(keys_h[i] for i in prng.choice(len(keys_h), size=int(round(POISON * len(keys_h))), replace=False)) if POISON > 0 and keys_h else set()
    for k in pois:
        cells[k]["E"] += 450
    rat = [c["E"] / (1 + len(c["conv"])) for (t, d), c in cells.items() if d in hd]
    for q in QS:
        KQ[q] = float(np.quantile(rat, q)) if rat else 4.0
    print("kappa", {q: round(KQ[q], 2) for q in QS}, flush=True)
    hc = [c for (t, d), c in cells.items() if d in hd]
    KR["recv"] = float(np.quantile([c["E"] / (1 + len(c["recv"])) for c in hc], 0.9)) if hc else 4.0
    KR["recv_est"] = float(np.quantile([c["E"] / (1 + len(c["recv_est"])) for c in hc], 0.9)) if hc else 4.0
    KR["A"] = float(np.quantile([c["E"] / (1 + c["A"]) for c in hc], 0.9)) if hc else 4.0
    for q in (95, 99):
        KR[f"A{q}"] = float(np.quantile([c["E"] / (1 + c["A"]) for c in hc], q / 100)) if hc else 4.0
    print("kappa_recv", {k: round(v, 2) for k, v in KR.items()}, flush=True)
    for k in pois:
        cells[k]["E"] -= 450
    ACT = Counter()
    for c in cells.values():
        for j, k in c["resp"].items():
            ACT[j] += k
    ev = sorted(S["eval"])
    pols = ["edges", "responders", "eff_resp_D1", "eff_resp_D2", "est_resp", "est_eff_D1", "new_unit_D1",
            "new_unit_est_D1", "conv_resp", "conv_new_unit", "conv_w25", "conv_w50", "clip_k2", "clip_k4", "clip_k8", "clip_k16", "clipA_q50", "clipA_q75", "clipA_q90", "AxR_b25", "AxR_b50", "AxR_b75", "anchR_q90", "anchRE_q90", "AxRE_b75", "anchA_q90", "anchA_q95", "anchA_q99", "anchAR_q90", "anchAR_q99", "trim2", "trim1", "authors", "posts", "oracle", "ranker", "gate"]
    out = {}
    for name, (kind, B, R) in ATTACKS.items():
        rng = np.random.default_rng([20261107, B, R, len(kind)])
        tot = {p: [] for p in pols}
        days = {p: {} for p in pols}
        succ = {p: [] for p in pols}
        draws = 1 if kind == "none" else DRAWS
        for _ in range(draws):
            acc = {p: 0 for p in pols}
            dayv = {p: {} for p in pols}
            # pods: one fixed victim unit and one fixed group of established accounts for the whole evaluation
            pod_victim, pod_ids, pod_seen = None, None, set()
            for d in ev:
                T = [t for (t, dd) in cells if dd == d]
                if not T:
                    continue
                low = sorted(T, key=lambda t: len(cells[(t, d)]["T"]))[: max(1, len(T) // 2)]
                if VICTIM == "near":
                    dsc = sorted(T, key=lambda t: (-len(cells[(t, d)]["T"]), t))
                    low = dsc[3:6] or dsc[-1:]
                fake, fest, fids = [], [], []
                victim = None
                if kind == "new":
                    victim = low[rng.integers(len(low))]
                    fake = [max(1, R // B)] * B
                elif kind in ("sleeper_low", "sleeper_high", "sleeper_conv"):
                    victim = low[rng.integers(len(low))]
                    pool = sorted(cells[(victim, d)]["pool"], key=lambda j: (ACT.get(j, 0), j))
                    half = len(pool) // 2
                    pool = pool[:half] if kind == "sleeper_low" else pool[half:] if kind == "sleeper_high" else pool
                    if pool:
                        fids = list(rng.choice(pool, size=min(B, len(pool)), replace=False))
                        fest = [max(1, R // B)] * len(fids)
                elif kind in ("post_flood", "post_reply"):
                    victim = low[rng.integers(len(low))]
                    if kind == "post_reply":
                        fake = [max(1, R // B)] * B
                elif kind == "all_staged":
                    victim = low[rng.integers(len(low))]
                    fake = [max(1, R // B)] * B + [1] * B
                elif kind in ("r2r_new", "r2r_aged", "all_sides"):
                    victim = low[rng.integers(len(low))]
                    if kind in ("r2r_new", "all_sides"):
                        fake = [max(1, R // B)] * B
                    else:
                        pool = sorted(cells[(victim, d)]["pool"], key=lambda j: (ACT.get(j, 0), j))
                        pool = pool[len(pool) // 2:]
                        if pool:
                            fids = list(rng.choice(pool, size=min(B, len(pool)), replace=False))
                            fest = [max(1, R // B)] * len(fids)
                elif kind == "pod":
                    if pod_victim is None or (pod_victim, d) not in cells:
                        cand = [t for t in low]
                        pod_victim = cand[rng.integers(len(cand))]
                    victim = pod_victim if (pod_victim, d) in cells else None
                    if pod_ids is None:
                        pool = sorted(cells[(T[0], d)]["pool"])
                        pod_ids = list(rng.choice(pool, size=min(B, len(pool)), replace=False)) if pool else []
                    fids, fest = pod_ids, [max(1, R // max(1, B))] * len(pod_ids)
                if kind == "pod" and victim is not None:
                    # pod accounts already replied to this unit on earlier days: not new to the unit any more
                    c = cells[(victim, d)]
                    c_prev = c["prev_unit"]
                    c["prev_unit"] = c_prev | pod_seen
                if kind == "sleeper_conv" and victim is not None:
                    cv = cells[(victim, d)]
                    cv_conv = cv["conv"]
                    cv["conv"] = cv_conv | set(fids)
                    cv_resp = cv["resp"]
                    # staged exchanges among aged accounts: each fake reply is answered by another fake (established)
                    cv_recv, cv_recv_est = cv["recv"], cv["recv_est"]
                    cv["recv"] = cv_recv | set(fids)
                    cv["recv_est"] = cv_recv_est | set(fids)
                if kind == "all_staged" and victim is not None:
                    cs = cells[(victim, d)]
                    saved_s = (cs["conv"], cs["recv"], cs["recv_est"])
                    fk = {("fake", i) for i in range(2 * B)}
                    cs["conv"] = cs["conv"] | fk
                    cs["recv"] = cs["recv"] | {("fake", i) for i in range(B)}
                if kind in ("post_flood", "post_reply", "all_sides", "all_staged") and victim is not None:
                    cp = cells[(victim, d)]
                    cp["A"] += B; cp["V"] += B
                if kind in ("r2r_new", "r2r_aged", "all_sides") and victim is not None:
                    cv = cells[(victim, d)]
                    saved = (cv["conv"], cv["recv"], cv["recv_est"])
                    U = sorted(cv["d1u"])
                    hit = set(rng.choice(U, size=min(R, len(U)), replace=True)) if U else set()   # answered real repliers
                    nfake = B if kind in ("r2r_new", "all_sides") else 0
                    cv["conv"] = cv["conv"] | set(fids) | {("fake", i) for i in range(nfake)}
                    cv["recv"] = cv["recv"] | hit
                    if kind == "r2r_aged":
                        cv["recv_est"] = cv["recv_est"] | hit
                sc = {t: scores(cells[(t, d)], fake if t == victim else [], fest if t == victim else [],
                                fids if t == victim else []) for t in T}
                global T_DAY
                T_DAY = d
                nfp = B if (kind in ("post_flood", "post_reply", "all_sides", "all_staged") and victim is not None) else 0
                learned_scores(cells, T, sc, victim, (sum(fake) + sum(fest)) if victim is not None else 0, nfp)
                for t in T:
                    sc[t].pop("_gate_sig", None)
                if kind == "sleeper_conv" and victim is not None:
                    cv["conv"], cv["resp"] = cv_conv, cv_resp
                    cv["recv"], cv["recv_est"] = cv_recv, cv_recv_est
                if kind in ("r2r_new", "r2r_aged", "all_sides") and victim is not None:
                    cv["conv"], cv["recv"], cv["recv_est"] = saved
                if kind == "all_staged" and victim is not None:
                    cs["conv"], cs["recv"], cs["recv_est"] = saved_s
                if kind in ("post_flood", "post_reply", "all_sides", "all_staged") and victim is not None:
                    cp["A"] -= B; cp["V"] -= B
                if kind == "pod" and victim is not None:
                    cells[(victim, d)]["prev_unit"] = c_prev
                    pod_seen |= set(fids)
                for p in pols:
                    sel = sorted(T, key=lambda t: (-sc[t][p], t))[:3]
                    y_ = len(set().union(*[cells[(t, d)]["T"] for t in sel]))
                    acc[p] += y_
                    dayv[p][str(d)] = y_
                    if victim is not None:
                        succ[p].append(victim in sel)
            for p in pols:
                tot[p].append(acc[p])
                for k_, y_ in dayv[p].items():
                    days[p].setdefault(k_, []).append(y_)
        out[name] = {p: float(np.mean(v)) for p, v in tot.items()}
        out[name]["_succ"] = {p: (float(np.mean(v)) if v else None) for p, v in succ.items()}
        out[name]["_days"] = {p: {k_: float(np.mean(v_)) for k_, v_ in days[p].items()} for p in pols}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim", default="low")
    ap.add_argument("--place", default="top", choices=["top", "spread"])
    ap.add_argument("--only", default="", help="comma-separated settings (debug)")
    a = ap.parse_args()
    global VICTIM, PLACE, W_RK, CUR
    VICTIM, PLACE = a.victim, a.place
    SS = MECH.settings(False)
    names = [x for x in SS if not x.endswith("_terms")]
    # frozen ranker (www_ranker.py default mode)
    data = {n: (*RK.build(SS[n]), SS[n]) for n in RK.DEV_SETTINGS}
    W_RK = RK.train(data, RK.DEV_SETTINGS)
    print("ranker weights", dict(zip(RK.FEATS, map(float, W_RK))), flush=True)
    # gate training rows (www_gate.py): development days of the development settings
    grow = {}
    for n in RK.DEV_SETTINGS:
        S = SS[n]
        cells = data[n][0]
        grow[n] = []
        for d in sorted(S["dev"]):
            keys = sorted(k for k in cells if k[1] == d)
            if len(keys) < 4:
                continue
            M = np.array([gate_sig(cells[k]["V"], cells[k]["A"], cells[k]["e"].sum(), cells[k]["hist_y"], cells[k]["e"])
                          for k in keys], float)
            ys = []
            for j in range(len(SIGN)):
                order = sorted(range(len(keys)), key=lambda i: (-M[i, j], keys[i][0]))[:3]
                ys.append(len(set().union(*[cells[keys[i]]["T"] for i in order])))
            grow[n].append((gate_dayfeat(M, np.concatenate([cells[k]["e"] for k in keys])), np.array(ys, float)))
    folds = [["cyber_main", "cyber_fresh"], ["nba"], ["crypto"], ["ai"]]
    for n in names:
        held = next((f for f in folds if n in f), [])
        GATE[n] = gate_fit(grow, [m for m in RK.DEV_SETTINGS if m not in held])
    res = {}
    run_names = [x for x in names if not a.only or x in a.only.split(",")]
    for n in run_names:
        CUR = n
        res[n] = run(SS[n])
        print(n, "clean ranker", res[n]["clean"]["ranker"], "gate", res[n]["clean"]["gate"], flush=True)
    res["_n_small_days"] = dict(N_SMALL)
    res["_ranker_weights"] = dict(zip(RK.FEATS, map(float, W_RK)))
    tag = ("_spread" if a.place == "spread" else "") + ("" if a.victim == "low" else f"_{a.victim}") + ("_debug" if a.only else "")
    Path(f"outputs/interaction_experiment/www_eval/bench_attack{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
