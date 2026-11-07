"""LLM annotator 'A_llm' for the WWW audits T1-T4 (2026-11-07). Model: Qwen/Qwen3.5-4B (latest Qwen 4B, 2026-02),
greedy, thinking off. Same items and the same guideline sections (ANNOTATION_PLAN.md) the human annotators see; T3
without web search. Output: CSVs in the human export format under outputs/www_audit/returned/llm_A/ (post text:
outputs only). NOTE: the labels being audited (topics T1, victim names T2) come from Qwen3-4B, a model of the same
family; this rater is a model-agreement check, not an independent human audit.

    .venv-embed/bin/python scripts/www_llm_annotator.py [--tasks T1,T2,T3,T4] [--limit N]
"""
import argparse, csv, html, json, re, sys
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
AUD = Path("outputs/www_audit"); OUT = AUD / "returned/llm_A"
PLAN = Path("www2027/docs/annotation/ANNOTATION_PLAN.md").read_text()
TOPICS = ["Ransomware", "Vulnerability / CVE", "Data breach", "Phishing / scam", "Malware / infostealer",
          "Government advisory", "Policy / regulation", "Critical infrastructure", "DDoS / outage", "Financial fraud",
          "Cyber, none of these", "unsure"]
KEYS = dict(zip(TOPICS, ["ransomware", "vulnerability_cve", "data_breach", "phishing_scam", "malware_infostealer",
                         "gov_advisory", "policy_regulation", "critical_infrastructure", "ddos_outage", "fraud_financial",
                         "unassigned", "topic_unsure"]))  # the human page exports these keys

def section(start, end):
    i = PLAN.index(start); j = PLAN.index(end, i + 1)
    return PLAN[i:j]

def page_items(p):
    s = p.read_text()
    D = json.loads(re.search(r"const D=(\[.*?\]);const F=", s, re.S).group(1))
    F = json.loads(re.search(r"const F=(\[.*?\]);const KEY", s, re.S).group(1))
    strip = lambda h: html.unescape(re.sub(r"<[^>]+>", "\n", h)).strip()
    return [(d["id"], strip(d["html"])) for d in D], F

def tasks():
    T = {}
    rows = list(csv.DictReader(open(AUD / "topic/items.csv")))
    T["T1"] = dict(guide=section("## 四、T1", "## 五、T2"), items=[(r["item_id"], r["text"]) for r in rows],
                   schema='{"cyber": "yes|no|unsure", "topics": [zero or more of ' + json.dumps(TOPICS) + ']}',
                   cols=["item_id", "cyber", "topics"], out="topic_audit_A_llm.csv")
    for k, sub, a, b in [("T2", "extraction/extraction_A.html", "## 五、T2", "## 六、T3"),
                         ("T3", "matching/matching_A.html", "## 六、T3", "## 六之二、T4"),
                         ("T4", "domain/domain_A.html", "## 六之二、T4", "## 七、")]:
        items, F = page_items(AUD / sub)
        schema = "{" + ", ".join(f'"{f[0]}": ' + ('"' + "|".join(f[3]) + '"' if f[2] == "radio" else '"text, empty if none"')
                                 for f in F) + "}"
        T[k] = dict(guide=section(a, b), items=items, schema=schema, cols=["item_id"] + [f[0] for f in F],
                    fields=F, out=Path(sub).stem.replace("_A", "") + "_A_llm.csv")
    return T

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", default="T3,T4,T2,T1"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0").eval()
    OUT.mkdir(parents=True, exist_ok=True)
    T = tasks()
    for k in a.tasks.split(","):
        t = T[k]; items = t["items"][: a.limit] if a.limit else t["items"]
        sysmsg = ("You are an annotator for a research audit. Follow the guideline exactly (it is in Chinese; answer "
                  "options are in English). Judge only from the text shown; no web search.\n\n" + t["guide"])
        rows, raw = [], []
        for n, (iid, text) in enumerate(items):
            msgs = [{"role": "system", "content": sysmsg},
                    {"role": "user", "content": f"Item {iid}:\n{text}\n\nAnswer with one JSON object only: {t['schema']}"}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          enable_thinking=False, return_dict=True).to("cuda:0")
            with torch.no_grad():
                o = model.generate(**ids, max_new_tokens=160, do_sample=False)
            ans = tok.decode(o[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"\{.*\}", ans, re.S)
            try:
                j = json.loads(m.group(0)) if m else {}
            except Exception:
                j = {}
            if k == "T1":
                tp = j.get("topics") or []
                rows.append([iid, j.get("cyber", ""), ";".join(KEYS[x] for x in tp if x in KEYS) if j.get("cyber") == "yes" else ""])
            else:
                rows.append([iid] + [str(j.get(f[0], "")) for f in t["fields"]])
            raw.append({"item_id": iid, "answer": ans})
            if n % 50 == 0:
                print(k, n, len(items), rows[-1], flush=True)
        with open(OUT / t["out"], "w", newline="") as f:
            w = csv.writer(f); w.writerow(t["cols"]); w.writerows(rows)
        (OUT / (t["out"][:-4] + "_raw.jsonl")).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in raw))
        print("done", k, len(rows), flush=True)

if __name__ == "__main__":
    main()
