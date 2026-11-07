"""PREREG_DOSE3 tests T1-T3 on the six test domains, from outputs/.../manipulation.json (run after collection)."""
import json, numpy as np
d = json.load(open("outputs/interaction_experiment/www_eval/manipulation.json"))
H = [n for n in ["music", "books", "f1", "tennis", "health", "food"] if n in d]
v = lambda x: float(np.mean(x)) if isinstance(x, list) else float(x)
atk = [a for a in d[H[0]] if a != "clean" and not a.startswith("A8")]
S = lambda p, a: sum(v(d[n][a][p]) for n in H)
out = {"domains": H}
R, A = S("edges", "clean"), S("authors", "clean")
lossR = {a: 1 - S("edges", a) / R for a in atk}
for m in ["clipA_q90", "AxR_b75"]:
    cl = S(m, "clean")
    t1 = cl >= 0.97 * (R if m == "clipA_q90" else max(R, A))
    loss = {a: 1 - S(m, a) / cl for a in atk}
    t2 = all(loss[a] < 0.5 * lossR[a] for a in atk)
    worst = min(atk, key=lambda a: S(m, a))
    r = {"clean": cl, "clean_vs_R": cl / R, "clean_vs_A": cl / A, "T1": t1, "T2": t2, "loss": loss,
         "worst_attack": worst, "worst_yield": S(m, worst), "A8_loss": 1 - S(m, "A8_fake_conversation_B100_R200") / cl,
         "reading": "supported" if t1 and t2 else "partly" if t1 or t2 else "not supported",
         "per_domain_clean_vs_R": {n: v(d[n]["clean"][m]) / v(d[n]["clean"]["edges"]) for n in H}}
    if m == "AxR_b75":
        r["T3"] = S(m, worst) >= min(S("edges", a) for a in atk) and S(m, worst) >= A
        r["T3_detail"] = {"M2_worst": S(m, worst), "R_worst": min(S("edges", a) for a in atk), "A_clean": A}
    out[m] = r
out["edges"] = {"clean": R, "loss": lossR}; out["authors_clean"] = A
print(json.dumps(out, indent=1))
open("outputs/interaction_experiment/www_eval/dose3_tests.json", "w").write(json.dumps(out, indent=1))
