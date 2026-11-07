"""Synthetic engagement manipulation (stress test for listener-side counts), paper 2.

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

sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("mech", "scripts/www_dose2_mech.py")
MECH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MECH)
MON = MECH.MON
ACT = Counter()
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
                         "T": after - before, "conv": conv}
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
               for k in (2, 4, 8, 16)}}


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
}


def run(S):
    global ACT
    cells = cells_of(S)
    ACT = Counter()
    for c in cells.values():
        for j, k in c["resp"].items():
            ACT[j] += k
    ev = sorted(S["eval"])
    pols = ["edges", "responders", "eff_resp_D1", "eff_resp_D2", "est_resp", "est_eff_D1", "new_unit_D1",
            "new_unit_est_D1", "conv_resp", "conv_new_unit", "conv_w25", "conv_w50", "clip_k2", "clip_k4", "clip_k8", "clip_k16", "authors", "posts", "oracle"]
    out = {}
    for name, (kind, B, R) in ATTACKS.items():
        rng = np.random.default_rng([20261107, B, R, len(kind)])
        tot = {p: [] for p in pols}
        draws = 1 if kind == "none" else DRAWS
        for _ in range(draws):
            acc = {p: 0 for p in pols}
            # pods: one fixed victim unit and one fixed group of established accounts for the whole evaluation
            pod_victim, pod_ids, pod_seen = None, None, set()
            for d in ev:
                T = [t for (t, dd) in cells if dd == d]
                if not T:
                    continue
                low = sorted(T, key=lambda t: len(cells[(t, d)]["T"]))[: max(1, len(T) // 2)]
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
                sc = {t: scores(cells[(t, d)], fake if t == victim else [], fest if t == victim else [],
                                fids if t == victim else []) for t in T}
                if kind == "sleeper_conv" and victim is not None:
                    cv["conv"], cv["resp"] = cv_conv, cv_resp
                if kind == "pod" and victim is not None:
                    cells[(victim, d)]["prev_unit"] = c_prev
                    pod_seen |= set(fids)
                for p in pols:
                    sel = sorted(T, key=lambda t: (-sc[t][p], t))[:3]
                    acc[p] += len(set().union(*[cells[(t, d)]["T"] for t in sel]))
            for p in pols:
                tot[p].append(acc[p])
        out[name] = {p: float(np.mean(v)) for p, v in tot.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-only", action="store_true")
    a = ap.parse_args()
    SS = MECH.settings(a.known_only)
    res = {}
    for n in [x for x in SS if not x.endswith("_terms")]:
        res[n] = run(SS[n])
        print(n, flush=True)
        for k, r in res[n].items():
            print("  ", k, {p: round(v) for p, v in r.items()}, flush=True)
    tag = "_known" if a.known_only else ""
    Path(f"outputs/interaction_experiment/www_eval/manipulation{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
