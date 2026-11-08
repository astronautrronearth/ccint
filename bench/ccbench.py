"""ccbench: clean utility and manipulation sensitivity of daily topic selection (v1, 2026-11-08).

Task. Each day, at a fixed cutoff, a monitor picks k = 3 topics from what it can observe up to the cutoff. Score:
the number of distinct accounts that first engage with any of the three topics in the 24 h after the cutoff
(union of next-day new responders), as a share of the best possible pick (oracle).
Every method is scored twice on the same decisions: on clean data and under a fixed suite of simulated
manipulation attacks, where an attacker lifts one low-yield victim topic per day with fake accounts. Fake accounts
never count as genuine new responders, so the target is unchanged. Reported per tier (dev / held / test):
clean yield, worst yield over the partial attacks A1-A11 and over all attacks A1-A13, and the share of attacked
days on which the strongest attack puts its victim into the top 3.

Data: one JSON per setting (scripts/www_bench_export.py). Only numpy and scipy are needed.

Writing a method:
    class MyMethod(Method):
        name = "my"
        def calibrate(self, history):          # optional; history = {day: {topic: View}} on the setting's history days
            ...
        def fit(self, train):                  # optional; learned methods; train = [(views, targets)] dev days
            ...
        def score(self, views):                # views = {topic: View} for one day (possibly attacked)
            return {t: v.replies for t, v in views.items()}
    res = evaluate([MyMethod()], load("outputs/ccbench_v1"))
    print(summarize(res))

A View exposes only what is observable at the cutoff: posts (V), authors (A), replies (R, replies+quotes before the
cutoff), per_post (array of replies per post), responders ({account: replies}), conversational (set), answered
(depth-1 repliers answered by another account), answered_established (answered by an account seen on an earlier
day), established ({account: replies} by accounts seen on an earlier day), prev_responders (accounts that replied to
this topic on earlier days), history_yield (mean next-day new responders of this topic on earlier days).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

DEV = ["cyber_main", "cyber_fresh", "nba", "crypto", "ai"]
FOLDS = [["cyber_main", "cyber_fresh"], ["nba"], ["crypto"], ["ai"]]   # leave-one-setting-out for dev settings
DRAWS = 20

# name: (kind, B accounts, R fake replies); parameters from measured campaigns (see the paper)
ATTACKS = {
    "clean": ("none", 0, 0),
    "A1_reply_bomb_B5_R200": ("new", 5, 200), "A1_reply_bomb_B5_R1000": ("new", 5, 1000),
    "A2_sybil_B300_R450": ("new", 300, 450), "A2_sybil_B1000_R1500": ("new", 1000, 1500),
    "A3_sleeper_B100_R150": ("sleeper_low", 100, 150), "A3_sleeper_B500_R750": ("sleeper_low", 500, 750),
    "A4_compromised_B100_R100": ("sleeper_high", 100, 100), "A4_compromised_B500_R500": ("sleeper_high", 500, 500),
    "A5_pod_G50_R50": ("pod", 50, 50), "A5_pod_G350_R350": ("pod", 350, 350),
    "A6_sockpuppet_B3_R15": ("new", 3, 15),
    "A7_brigade_B200_R400": ("sleeper_low", 200, 400),
    "A8_fake_conversation_B100_R200": ("sleeper_conv", 100, 200),
    "A9a_reply_to_replies_B300_R450": ("r2r_new", 300, 450),
    "A9b_reply_to_replies_aged_B300_R450": ("r2r_aged", 300, 450),
    "A10_post_flood_B300": ("post_flood", 300, 0),
    "A11_post_and_reply_B300_R450": ("post_reply", 300, 450),
    "A12_all_sides_B30_R45": ("all_sides", 30, 45), "A12_all_sides_B100_R150": ("all_sides", 100, 150),
    "A12_all_sides_B300_R450": ("all_sides", 300, 450),
    "A13_staged_all_B30_R45": ("all_staged", 30, 45), "A13_staged_all_B100_R150": ("all_staged", 100, 150),
    "A13_staged_all_B300_R450": ("all_staged", 300, 450),
}
PARTIAL = [a for a in ATTACKS if a != "clean" and not a.startswith(("A12", "A13"))]
ALL_ATTACKS = [a for a in ATTACKS if a != "clean"]


@dataclass
class View:
    posts: int
    authors: int
    replies: float
    per_post: np.ndarray
    responders: Counter
    conversational: set
    answered: set
    answered_established: set
    established: Counter
    prev_responders: set
    history_yield: float


@dataclass
class Cell:
    V: int
    A: int
    E: float
    e: np.ndarray
    resp: Counter
    conv: set
    recv_by: dict
    d1u: set
    T: set
    prev_unit: set = field(default_factory=set)
    recv: set = field(default_factory=set)
    recv_est: set = field(default_factory=set)
    est: Counter = field(default_factory=Counter)
    pool: set = field(default_factory=set)
    hist_y: float = 0.0


@dataclass
class Setting:
    name: str
    tier: str
    hist: list
    dev: list
    eval: list
    cells: dict        # (topic, day) -> Cell, in the published order (victim tie-breaks depend on it)
    act: Counter


def _setting(doc):
    cells = {}
    for r in doc["cells"]:
        e = np.array([k for _, k in r["posts"]], float)
        cells[(r["topic"], r["day"])] = Cell(
            V=len(r["posts"]), A=len({a for a, _ in r["posts"]}), E=float(e.sum()), e=e,
            resp=Counter({int(j): k for j, k in r["responders"].items()}), conv=set(r["conversational"]),
            recv_by={int(j): set(v) for j, v in r["answered_by"].items()}, d1u=set(r["unanswered"]), T=set(r["target"]))
    first_seen, seen_unit = {}, {}
    for (t, d), c in sorted(cells.items(), key=lambda kv: kv[0][1]):
        for j in c.resp:
            first_seen.setdefault(j, d)
    for (t, d), c in sorted(cells.items(), key=lambda kv: kv[0][1]):
        prev = seen_unit.setdefault(t, set())
        c.prev_unit = set(prev)
        prev |= set(c.resp)
    pool = {}
    for (t, d), c in cells.items():
        c.recv = set(c.recv_by)
        c.recv_est = {j for j, ks in c.recv_by.items() if any(first_seen.get(k, d) < d for k in ks)}
        c.est = Counter({j: k for j, k in c.resp.items() if first_seen.get(j, d) < d})
        pool.setdefault(d, set()).update(j for j in first_seen if first_seen[j] < d)
    by_topic = {}
    for (t, d), c in cells.items():
        by_topic.setdefault(t, {})[d] = c
    for (t, d), c in cells.items():
        c.pool = pool[d]
        prev = [x for dd, x in by_topic[t].items() if dd < d]
        c.hist_y = float(np.mean([len(x.T) for x in prev])) if prev else 0.0
    act = Counter()
    for c in cells.values():
        for j, k in c.resp.items():
            act[j] += k
    D = doc["days"]
    return Setting(doc["setting"], doc["tier"], D["hist"], D["dev"], D["eval"], cells, act)


def load(path):
    path = Path(path)
    splits = json.loads((path / "splits.json").read_text())
    return {n: _setting(json.loads((path / f"{n}.json").read_text())) for t in ("dev", "held", "test") for n in splits[t]}


def clean_view(c):
    return View(c.V, c.A, c.E, c.e, c.resp, c.conv, c.recv, c.recv_est, c.est, c.prev_unit, c.hist_y)


def day_views(S, d):
    return {t: clean_view(c) for (t, dd), c in S.cells.items() if dd == d}


class Method:
    name = "method"

    def calibrate(self, history):
        pass

    def fit(self, train):
        pass

    def score(self, views):
        raise NotImplementedError


# ---------------------------------------------------------------- reference methods (the paper's baselines)
class Count(Method):
    def __init__(self, name, f):
        self.name, self.f = name, f

    def score(self, views):
        return {t: float(self.f(v)) for t, v in views.items()}


def _q90(history, f):
    x = [f(v) for day in history.values() for v in day.values()]
    return float(np.quantile(x, 0.9)) if x else 4.0


class Capped(Method):
    """min(R, kA (1 + A), kC (1 + C)) with per-setting q90 caps from history; sides in {'A', 'C', 'conv'}."""

    def __init__(self, name, sides):
        self.name, self.sides = name, sides

    def calibrate(self, history):
        self.k = {"A": _q90(history, lambda v: v.replies / (1 + v.authors)),
                  "C": _q90(history, lambda v: v.replies / (1 + len(v.answered))),
                  "conv": _q90(history, lambda v: v.replies / (1 + len(v.conversational)))}

    def score(self, views):
        side = {"A": lambda v: v.authors, "C": lambda v: len(v.answered), "conv": lambda v: len(v.conversational)}
        return {t: float(min([v.replies] + [self.k[s] * (1 + side[s](v)) for s in self.sides])) for t, v in views.items()}


def _peak5(v):
    return np.sort(v.per_post)[::-1][:5].sum()


class Ranker(Method):
    """Ridge on within-day standardised log1p(R), log1p(peak5), log1p(history_yield); target next-day new
    responders demeaned within day (the paper's frozen ranker)."""
    name = "ranker"

    @staticmethod
    def _F(views):
        T = list(views)
        F = np.array([[np.log1p(views[t].per_post.sum()), np.log1p(_peak5(views[t])), np.log1p(views[t].history_yield)]
                      for t in T])
        return T, (F - F.mean(0)) / (F.std(0) + 1e-6)

    def fit(self, train):
        X, y = [], []
        for views, targets in train:
            if len(views) < 2:
                continue
            T, Z = self._F(views)
            tt = np.array([len(targets[t]) for t in T], float)
            X += list(Z)
            y += list(tt - tt.mean())
        X, y = np.array(X), np.array(y)
        self.w = np.linalg.solve(X.T @ X + 1.0 * np.eye(X.shape[1]), X.T @ y)

    def score(self, views):
        T, Z = self._F(views)
        return {t: float(z @ self.w) for t, z in zip(T, Z)}


class Gate(Method):
    """Per day, pick one of posts, authors, replies, history yield, peak5 from day-level statistics (ridge on the
    relative yield of each signal on development days)."""
    name = "gate"
    SIG = [lambda v: v.posts, lambda v: v.authors, lambda v: v.per_post.sum(), lambda v: v.history_yield, _peak5]

    def _M(self, views):
        T = list(views)
        return T, np.array([[f(views[t]) for f in self.SIG] for t in T], float)

    @staticmethod
    def _x(views, M):
        E = M[:, 2]
        allE = np.concatenate([v.per_post for v in views.values()])
        ag = []
        for i in range(M.shape[1]):
            for j in range(i + 1, M.shape[1]):
                r = spearmanr(M[:, i], M[:, j]).correlation
                ag.append(0.0 if np.isnan(r) else r)
        return np.array([E.max() / max(1, E.sum()), allE.max() / max(1, allE.sum()), np.log(len(M)),
                         np.log1p(M[:, 0].sum()), np.log1p(E.sum()), *ag, 1.0])

    def fit(self, train):
        X, Y = [], []
        for views, targets in train:
            if len(views) < 4:
                continue
            T, M = self._M(views)
            y = []
            for j in range(M.shape[1]):
                sel = sorted(range(len(T)), key=lambda i: (-M[i, j], T[i]))[:3]
                y.append(len(set().union(*[targets[T[i]] for i in sel])))
            y = np.array(y, float)
            X.append(self._x(views, M)); Y.append(y / max(1, y.max()))
        X, Y = np.array(X), np.array(Y)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.sd[-1], self.mu[-1] = 1, 0
        Xs = (X - self.mu) / self.sd
        self.W = np.linalg.solve(Xs.T @ Xs + 5 * np.eye(X.shape[1]), Xs.T @ Y)

    def score(self, views):
        T, M = self._M(views)
        j = int(np.argmax(((self._x(views, M) - self.mu) / self.sd) @ self.W)) if len(T) >= 4 else 2
        return {t: float(m[j]) for t, m in zip(T, M)}


def reference_methods():
    return [Count("replies", lambda v: v.replies), Count("posts", lambda v: v.posts),
            Count("authors", lambda v: v.authors), Count("conv_resp", lambda v: len(v.conversational)),
            Capped("clip_conv", ["conv"]), Capped("cap_answered", ["C"]), Capped("cap_authors", ["A"]),
            Capped("cross_anchored", ["A", "C"]), Ranker(), Gate()]


# ---------------------------------------------------------------- attack simulation
def _place(e, add, place):
    if add <= 0 or len(e) == 0:
        return e
    e = e.copy()
    if place == "top":
        e[int(np.argmax(e))] += add
    else:
        q, r = divmod(int(add), len(e))
        e += q
        order = np.argsort(-e - 1e-9 * np.arange(len(e)))
        e[order[:r]] += 1
    return e


def _attacked_views(S, d, T, kind, B, R, rng, state, victim_rule, place):
    cells = S.cells
    low = sorted(T, key=lambda t: len(cells[(t, d)].T))[: max(1, len(T) // 2)]
    if victim_rule == "near":
        dsc = sorted(T, key=lambda t: (-len(cells[(t, d)].T), t))
        low = dsc[3:6] or dsc[-1:]
    fake, fest, fids, victim = [], [], [], None
    if kind == "none":
        return None, {t: clean_view(cells[(t, d)]) for t in T}
    if kind == "new":
        victim = low[rng.integers(len(low))]
        fake = [max(1, R // B)] * B
    elif kind in ("sleeper_low", "sleeper_high", "sleeper_conv"):
        victim = low[rng.integers(len(low))]
        pool = sorted(cells[(victim, d)].pool, key=lambda j: (S.act.get(j, 0), j))
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
            pool = sorted(cells[(victim, d)].pool, key=lambda j: (S.act.get(j, 0), j))
            pool = pool[len(pool) // 2:]
            if pool:
                fids = list(rng.choice(pool, size=min(B, len(pool)), replace=False))
                fest = [max(1, R // B)] * len(fids)
    elif kind == "pod":
        if state["pod_victim"] is None or (state["pod_victim"], d) not in cells:
            state["pod_victim"] = low[rng.integers(len(low))]
        victim = state["pod_victim"]
        if state["pod_ids"] is None:
            pool = sorted(cells[(T[0], d)].pool)
            state["pod_ids"] = list(rng.choice(pool, size=min(B, len(pool)), replace=False)) if pool else []
        fids, fest = state["pod_ids"], [max(1, R // max(1, B))] * len(state["pod_ids"])
    c = cells[(victim, d)]
    conv, recv, recv_est, prev = set(c.conv), set(c.recv), set(c.recv_est), set(c.prev_unit)
    V, A = c.V, c.A
    if kind == "pod":
        prev |= state["pod_seen"]
    if kind == "sleeper_conv":
        conv |= set(fids); recv |= set(fids); recv_est |= set(fids)
    if kind == "all_staged":
        conv |= {("fake", i) for i in range(2 * B)}
        recv |= {("fake", i) for i in range(B)}
    if kind in ("post_flood", "post_reply", "all_sides", "all_staged"):
        A += B; V += B
    if kind in ("r2r_new", "r2r_aged", "all_sides"):
        U = sorted(c.d1u)
        hit = set(rng.choice(U, size=min(R, len(U)), replace=True)) if U else set()
        nfake = B if kind in ("r2r_new", "all_sides") else 0
        conv |= set(fids) | {("fake", i) for i in range(nfake)}
        recv |= hit
        if kind == "r2r_aged":
            recv_est |= hit
    if kind == "pod":
        state["pod_seen"] |= set(fids)
    add = sum(fake) + sum(fest)
    resp = Counter(c.resp)
    est = Counter(c.est)
    for i, k in enumerate(fake):
        resp[("new", i)] += k
    for j, k in zip(fids, fest):
        resp[j] += k
        est[j] += k
    e = _place(c.e, add, place)
    if kind in ("post_flood", "post_reply", "all_sides", "all_staged"):
        e = np.concatenate([e, np.zeros(B)])
    views = {t: clean_view(cells[(t, d)]) for t in T}
    views[victim] = View(V, A, c.E + add, e, resp, conv, recv, recv_est, est, prev, c.hist_y)
    return victim, views


def _train_set(settings, names):
    out = []
    for n in names:
        S = settings[n]
        for d in sorted(S.dev):
            v = day_views(S, d)
            out.append((v, {t: S.cells[(t, d)].T for t in v}))
    return out


def evaluate(methods, settings, victim="low", place="top", attacks=None, draws=DRAWS, k=3):
    """Returns {setting: {attack: {method: {"yield", "oracle", "days": {day: y}, "succ": share}}}}."""
    attacks = attacks or list(ATTACKS)
    res = {}
    for n, S in settings.items():
        held = next((f for f in FOLDS if n in f), [])
        for m in methods:
            m.calibrate({d: day_views(S, d) for d in S.hist})
            m.fit(_train_set(settings, [x for x in DEV if x not in held and x in settings]))
        res[n] = {}
        for a in attacks:
            kind, B, R = ATTACKS[a]
            rng = np.random.default_rng([20261107, B, R, len(kind)])
            acc = {m.name: [] for m in methods}
            days = {m.name: {} for m in methods}
            succ = {m.name: [] for m in methods}
            ora = []
            for _ in range(1 if kind == "none" else draws):
                state = {"pod_victim": None, "pod_ids": None, "pod_seen": set()}
                tot = {m.name: 0 for m in methods}
                o = 0
                for d in sorted(S.eval):
                    T = [t for (t, dd) in S.cells if dd == d]
                    if not T:
                        continue
                    vic, views = _attacked_views(S, d, T, kind, B, R, rng, state, victim, place)
                    o += len(set().union(*[S.cells[(t, d)].T for t in sorted(T, key=lambda t: (-len(S.cells[(t, d)].T), t))[:k]]))
                    for m in methods:
                        sc = m.score(views)
                        sel = sorted(T, key=lambda t: (-sc[t], t))[:k]
                        y = len(set().union(*[S.cells[(t, d)].T for t in sel]))
                        tot[m.name] += y
                        days[m.name].setdefault(d, []).append(y)
                        if vic is not None:
                            succ[m.name].append(vic in sel)
                for m in methods:
                    acc[m.name].append(tot[m.name])
                ora.append(o)
            res[n][a] = {m.name: {"yield": float(np.mean(acc[m.name])), "oracle": float(np.mean(ora)),
                                  "days": {d: float(np.mean(v)) for d, v in days[m.name].items()},
                                  "succ": float(np.mean(succ[m.name])) if succ[m.name] else None} for m in methods}
    return res


def summarize(res, settings):
    """Per tier: clean, worst over A1-A11, worst over A1-A13 (% of oracle summed over the tier's settings) and the
    strongest partial attack's success rate (pooled over attacked days)."""
    out = {}
    tiers = {}
    for n, S in settings.items():
        tiers.setdefault(S.tier, []).append(n)
    methods = list(next(iter(res.values()))["clean"])
    for g, names in tiers.items():
        den = sum(res[n]["clean"][methods[0]]["oracle"] for n in names)
        pct = lambda m, a: 100 * sum(res[n][a][m]["yield"] for n in names) / den
        w = np.array([len(settings[n].eval) for n in names], float)
        out[g] = {}
        for m in methods:
            part = [a for a in PARTIAL if a in res[names[0]]]
            allA = [a for a in ALL_ATTACKS if a in res[names[0]]]
            s1 = max((float(np.dot([res[n][a][m]["succ"] for n in names], w) / w.sum()), a) for a in part) if part else (None, None)
            out[g][m] = {"clean": round(pct(m, "clean"), 1),
                         "worst_partial": round(min(pct(m, a) for a in part), 1) if part else None,
                         "worst_all": round(min(pct(m, a) for a in allA), 1) if allA else None,
                         "success_strongest_partial": round(100 * s1[0], 1) if s1[0] is not None else None,
                         "strongest_partial": s1[1]}
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--victim", default="low")
    ap.add_argument("--place", default="top")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    S = load(a.data)
    r = evaluate(reference_methods(), S, victim=a.victim, place=a.place)
    s = summarize(r, S)
    print(json.dumps(s, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": s, "raw": r}, default=str))
