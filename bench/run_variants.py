"""Run the 19 candidate methods (extended_methods) on one variant: top | top2 (= top, with A14) | boundary | spread | near | k1 | k5.
    python bench/run_variants.py top2 → outputs/interaction_experiment/www_eval/ccbench_ext_<variant>.json"""
import json, sys, time
sys.path.insert(0, "bench")
import ccbench as cb
S = cb.load("outputs/ccbench_v1")
tag = sys.argv[1]
kw = {"top": {}, "top2": {}, "boundary": {"victim": "boundary"}, "spread": {"place": "spread"}, "near": {"victim": "near"}, "k1": {"k": 1}, "k5": {"k": 5}}[tag]
t0 = time.time()
R = cb.evaluate(cb.extended_methods(), S, **kw)
s = cb.summarize(R, S)
json.dump({"summary": s, "raw": R}, open(f"outputs/interaction_experiment/www_eval/ccbench_ext_{tag}.json", "w"), default=str)
print(tag, round(time.time() - t0), "s")
for g in ("dev", "held", "test"):
    print(g)
    for m, r in s[g].items():
        print("  ", m.ljust(15), r["clean"], r["worst_partial"], r["worst_all"], r["success_strongest_partial"], r["strongest_partial"])
