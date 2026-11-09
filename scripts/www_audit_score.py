"""Score the human audits T1-T4 (paper 1) from annotator B's labels (2026-11-09).

Only annotator B (an author) returned labels; B labelled the random double-label subsets (T1 120, T2 50, T3 60,
T4 50). Following ANNOTATION_PLAN.md section 9 (fallback), B's labels estimate precision and the model annotator
A_llm (Qwen3.5-4B, outputs/www_audit/returned/llm_A) is the second rater for Cohen's kappa on the same items.
"unsure" (or blank) labels are excluded from precision denominators and reported; kappa is computed on items where
both raters gave a definite label. 95% Wilson intervals for unweighted shares.

    python scripts/www_audit_score.py → outputs/www_audit/scoring/audit_scores.json
"""
import csv
import json
import math
from collections import Counter
from pathlib import Path

A = Path("outputs/www_audit")
R = A / "returned"


def rd(p):
    with open(p, newline="", encoding="utf-8") as f:
        return {r["item_id"]: r for r in csv.DictReader(f)}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def kappa(a, b):
    """Cohen's kappa over paired definite labels."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return round((po - pe) / (1 - pe), 2) if pe < 1 else 1.0


def agree(B, L, f, ok=lambda v: v not in ("", "unsure")):
    ids = [i for i in B if i in L and ok(B[i][f]) and ok(L[i][f])]
    a, b = [B[i][f] for i in ids], [L[i][f] for i in ids]
    return {"n": len(ids), "agreement": round(100 * sum(x == y for x, y in zip(a, b)) / max(1, len(ids)), 1),
            "kappa": kappa(a, b)}


out = {}

# ---------------------------------------------------------------- T1 topic audit
key = rd(A / "topic/key.csv")
B = rd(R / "topic_audit_B.csv")
L = rd(R / "llm_A/topic_audit_A_llm.csv")
acc = [i for i in B if key[i]["stratum"] != "rejected"]
rej = [i for i in B if key[i]["stratum"] == "rejected"]
defin = [i for i in acc if B[i]["cyber"] in ("yes", "no")]
w = {i: float(key[i]["weight"]) for i in acc}
yes = [i for i in defin if B[i]["cyber"] == "yes"]
wp = sum(w[i] for i in yes) / sum(w[i] for i in defin)
# topic precision: the system topic is among B's topics (posts B judged cyber)
top_ok = [i for i in yes if key[i]["system_topic"] in B[i]["topics"].split(";")]
wt = sum(w[i] for i in top_ok) / sum(w[i] for i in yes)
rej_def = [i for i in rej if B[i]["cyber"] in ("yes", "no")]
out["T1"] = {
    "n_items": len(B), "n_accepted": len(acc), "n_rejected_stratum": len(rej),
    "unsure_or_blank": sum(B[i]["cyber"] not in ("yes", "no") for i in B),
    "cyber_precision_weighted": round(100 * wp, 1), "cyber_precision_unweighted": round(100 * len(yes) / len(defin), 1),
    "cyber_precision_ci_unweighted": wilson(len(yes), len(defin)), "n_cyber_definite": len(defin),
    "topic_precision_weighted": round(100 * wt, 1), "topic_precision_unweighted": round(100 * len(top_ok) / len(yes), 1),
    "topic_precision_ci_unweighted": wilson(len(top_ok), len(yes)), "n_topic": len(yes),
    "rejected_judged_cyber": [sum(B[i]["cyber"] == "yes" for i in rej_def), len(rej_def)],
    "kappa_cyber_B_vs_llm": agree(B, L, "cyber"),
    "topic_errors": Counter(f"{key[i]['system_topic']}->{B[i]['topics']}" for i in yes if i not in top_ok).most_common(8),
}
# kappa on topic correctness (system topic among the rater's topics), items both judged cyber
both = [i for i in acc if B[i]["cyber"] == "yes" and L.get(i, {}).get("cyber") == "yes"]
ta = ["y" if key[i]["system_topic"] in B[i]["topics"].split(";") else "n" for i in both]
tb = ["y" if key[i]["system_topic"] in L[i]["topics"].split(";") else "n" for i in both]
out["T1"]["kappa_topic_correct_B_vs_llm"] = {"n": len(both), "agreement": round(100 * sum(x == y for x, y in zip(ta, tb)) / max(1, len(both)), 1),
                                             "kappa": kappa(ta, tb)}

# ---------------------------------------------------------------- T2 victim extraction
key = rd(A / "extraction/key.csv")
B = rd(R / "extraction_B.csv")
L = rd(R / "llm_A/extraction_A_llm.csv")
lab = {i: B[i]["extraction"] for i in B}
# "none extracted, correctly" on a post where the model did extract a name is self-contradictory (E011, E050);
# treated as unsure (excluded) until the annotator confirms; listed in the output
CONTRA = [i for i in B if lab[i] == "none extracted, correctly" and key[i]["extracted"] != "(none extracted)"]
for i in CONTRA:
    lab[i] = "unsure"
has = [i for i in B if key[i]["extracted"] != "(none extracted)"]
none = [i for i in B if key[i]["extracted"] == "(none extracted)"]
hd = [i for i in has if lab[i] != "unsure"]
nd = [i for i in none if lab[i] != "unsure"]
correct = [i for i in hd if lab[i] == "all correct"]
partly = [i for i in hd if lab[i] == "partly correct"]
# a post has a true victim if B accepted at least one extracted name, or flagged a miss
true_victim = [i for i in hd + nd if lab[i] in ("all correct", "partly correct") or B[i]["missed"].strip()
               or (i in nd and lab[i] == "wrong")]
found = [i for i in true_victim if lab[i] in ("all correct", "partly correct")]
out["T2"] = {
    "n_items": len(B), "n_with_extraction": len(has), "n_without": len(none),
    "unsure": sum(lab[i] == "unsure" for i in B),
    "precision_all_correct": round(100 * len(correct) / max(1, len(hd)), 1), "precision_ci": wilson(len(correct), len(hd)),
    "partly_correct": len(partly), "n_precision": len(hd),
    "empty_correct": round(100 * sum(lab[i] == "none extracted, correctly" for i in nd) / max(1, len(nd)), 1), "n_empty": len(nd),
    "recall_posts": round(100 * len(found) / max(1, len(true_victim)), 1), "recall_ci": wilson(len(found), len(true_victim)),
    "n_true_victim_posts": len(true_victim),
    "contradictory_labels_treated_unsure": CONTRA,
    "kappa_B_vs_llm": agree({i: {"extraction": lab[i]} for i in B}, L, "extraction"),
}

# ---------------------------------------------------------------- T3 registry matching
key = rd(A / "matching/key.csv")
B = rd(R / "matching_B.csv")
L = rd(R / "llm_A/matching_A_llm.csv")
t3 = {"n_items": len(B), "unsure": sum(B[i]["same"] == "unsure" for i in B)}
for mt in ("exact", "fuzzy", "all"):
    ids = [i for i in B if (mt == "all" or key[i]["match_type"] == mt) and B[i]["same"] in ("yes", "no")]
    k = sum(B[i]["same"] == "yes" for i in ids)
    t3[f"precision_{mt}"] = round(100 * k / max(1, len(ids)), 1)
    t3[f"precision_{mt}_ci"] = wilson(k, len(ids))
    t3[f"n_{mt}"] = len(ids)
    t3[f"unsure_{mt}"] = sum(B[i]["same"] == "unsure" for i in B if mt == "all" or key[i]["match_type"] == mt)
fz = [i for i in B if key[i]["match_type"] == "fuzzy"]
t3["precision_fuzzy_unsure_as_wrong"] = round(100 * sum(B[i]["same"] == "yes" for i in fz) / len(fz), 1)
t3["n_fuzzy_all"] = len(fz)
t3["kappa_B_vs_llm"] = agree(B, L, "same")
t3["llm_label_counts"] = dict(Counter(L[i]["same"] for i in B))
t3["unsure_items"] = [(i, key[i]["match_type"], key[i]["candidate"], key[i]["registry"]) for i in B if B[i]["same"] == "unsure"]
out["T3"] = t3

# ---------------------------------------------------------------- T4 domain precision
key = rd(A / "domain/key.csv")
B = rd(R / "domain_B.csv")
L = rd(R / "llm_A/domain_A_llm.csv")
t4 = {"n_items": len(B), "unsure": sum(B[i]["in_domain"] not in ("yes", "no") for i in B)}
for dom in sorted({key[i]["domain"] for i in B}) + ["all"]:
    ids = [i for i in B if (dom == "all" or key[i]["domain"] == dom) and B[i]["in_domain"] in ("yes", "no")]
    k = sum(B[i]["in_domain"] == "yes" for i in ids)
    t4[f"precision::{dom}"] = [round(100 * k / max(1, len(ids)), 1), len(ids), wilson(k, len(ids))]
t4["kind_of_in_domain"] = dict(Counter(B[i]["kind"] or "blank" for i in B if B[i]["in_domain"] == "yes"))
t4["kappa_B_vs_llm"] = agree(B, L, "in_domain")
out["T4"] = t4

Path(A / "scoring").mkdir(exist_ok=True)
(A / "scoring/audit_scores.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
