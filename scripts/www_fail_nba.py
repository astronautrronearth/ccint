"""Failure-case inspection on the NBA development setting: per evaluation day, oracle vs edges picks with features."""
import importlib.util, sys
import numpy as np
sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("bf", "scripts/www_budget_fetch.py"); BF = importlib.util.module_from_spec(spec); spec.loader.exec_module(BF)
BF.ECAP = 10 ** 9
S = BF.MECH.settings(True)["nba"]
cells = BF.prepare(S, {}, {})
for d in sorted(S["eval"]):
    T = [t for (t, dd) in cells if dd == d]
    O = sorted(T, key=lambda t: -len(cells[(t, d)]["T"]))[:3]
    E = sorted(T, key=lambda t: -cells[(t, d)]["e"].sum())[:3]
    f = lambda t: f"{t}(y={len(cells[(t, d)]['T'])},V={cells[(t, d)]['V']},E={int(cells[(t, d)]['e'].sum())},top={int(cells[(t, d)]['e'].max()) if cells[(t, d)]['V'] else 0},h={cells[(t, d)]['hist_y']:.0f})"
    print(d, "ORACLE", [f(t) for t in O], "| EDGES", [f(t) for t in E])
