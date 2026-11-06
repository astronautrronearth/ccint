"""E-F topic-label audit: stratified sample and offline annotation pages (PREREG: www2027/results/frozen_eval).

    python scripts/www_topic_audit.py
→ outputs/www_audit/topic/  (contains post text; not committed, same rule as eval/)
    items.csv            item_id, text only (what annotators see)
    key.csv              item_id -> post_id, stratum, sampling weight, system topic (hidden from annotators)
    annotate_A.html      annotator A: all items
    annotate_B.html      annotator B: the 120-item double-label subset (independent; no discussion before done)

Strata: topic_v2s topic (11) x producer activity (author's P_scale posts: 1, 2-4, 5+) x template (text hash seen
>= 3 times in P_scale). 400 posts allocated equally over topics, then over the non-empty cells of each topic;
plus 80 posts from the rejected pool (rules_v2 not cyber, same window, English). Seed 20261106.
"""
from __future__ import annotations

import csv
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path[:0] = ["src"]
from ccint import db  # noqa: E402
from ccint.interaction import manifest as mf  # noqa: E402

OUT = Path("outputs/www_audit/topic")
TOPICS = [("ransomware", "Ransomware"), ("vulnerability_cve", "Vulnerability / CVE"), ("data_breach", "Data breach"),
          ("phishing_scam", "Phishing / scam"), ("malware_infostealer", "Malware / infostealer"),
          ("gov_advisory", "Government advisory"), ("policy_regulation", "Policy / regulation"),
          ("critical_infrastructure", "Critical infrastructure"), ("ddos_outage", "DDoS / outage"),
          ("fraud_financial", "Financial fraud"), ("unassigned", "Cyber, none of these")]
rnd = random.Random(20261106)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = mf.load("www_scale")
    ids = [int(r["post_id"]) for r in m.population("in_scale")]
    with db.connect() as c:
        rows = c.execute("""SELECT p.post_id, p.author_id, p.text, p.text_hash, t.topic_key FROM posts p
                            JOIN post_labels t ON t.post_id = p.post_id AND t.label_version = 'topic_v2s'
                            WHERE p.post_id = ANY(%s)""", (ids,)).fetchall()
        rej = c.execute("""SELECT p.post_id, p.text FROM posts p JOIN post_labels l ON l.post_id = p.post_id
                           AND l.label_version = 'rules_v2' WHERE NOT l.is_cyber AND p.source_key = 'bluesky'
                           AND p.lang LIKE 'en%%' AND length(coalesce(p.text, '')) > 20
                           AND p.published_at >= '2026-08-22' AND p.published_at < '2026-09-24'""").fetchall()
    na = Counter(r["author_id"] for r in rows)
    nh = Counter(r["text_hash"] for r in rows if r["text_hash"])
    act = lambda r: "1" if na[r["author_id"]] == 1 else ("2-4" if na[r["author_id"]] <= 4 else "5+")
    tpl = lambda r: "template" if r["text_hash"] and nh[r["text_hash"]] >= 3 else "single"
    cells = defaultdict(list)
    for r in rows:
        if r["topic_key"] in dict(TOPICS):
            cells[(r["topic_key"], act(r), tpl(r))].append(r)
    picked = []
    per_topic = 400 // len(TOPICS)
    extra = 400 - per_topic * len(TOPICS)
    for i, (t, _) in enumerate(TOPICS):
        tc = [k for k in cells if k[0] == t]
        quota = per_topic + (1 if i < extra else 0)
        alloc = {k: 0 for k in tc}
        left = quota
        while left > 0 and any(alloc[k] < len(cells[k]) for k in tc):
            for k in tc:
                if left and alloc[k] < len(cells[k]):
                    alloc[k] += 1
                    left -= 1
        for k, n in alloc.items():
            seen_hash, chosen = set(), []
            pool = cells[k][:]
            rnd.shuffle(pool)
            for r in pool:   # one post per text hash, so a template is labelled once per cell
                if len(chosen) >= n:
                    break
                if r["text_hash"] in seen_hash:
                    continue
                seen_hash.add(r["text_hash"])
                chosen.append(r)
            for r in chosen:
                picked.append({"post_id": r["post_id"], "text": r["text"], "stratum": "|".join(k),
                               "weight": round(len(cells[k]) / max(len(chosen), 1), 3), "system_topic": t})
    for r in rnd.sample(rej, 80):
        picked.append({"post_id": r["post_id"], "text": r["text"], "stratum": "rejected", "weight": None,
                       "system_topic": "not_cyber"})
    rnd.shuffle(picked)
    for i, r in enumerate(picked, 1):
        r["item_id"] = f"T{i:03d}"
    double = set(r["item_id"] for r in rnd.sample(picked, 120))
    with open(OUT / "items.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "text"])
        w.writerows([[r["item_id"], r["text"]] for r in picked])
    with open(OUT / "key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "post_id", "stratum", "weight", "system_topic", "double"])
        w.writerows([[r["item_id"], r["post_id"], r["stratum"], r["weight"], r["system_topic"], r["item_id"] in double]
                     for r in picked])
    page(OUT / "annotate_A.html", "A", [r for r in picked])
    page(OUT / "annotate_B.html", "B", [r for r in picked if r["item_id"] in double])
    meta = {"n": len(picked), "n_double": len(double), "strata": dict(Counter(r["stratum"] for r in picked))}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"n": meta["n"], "n_double": meta["n_double"]}))


def page(path, who, items):
    data = json.dumps([{"id": r["item_id"], "t": r["text"]} for r in items], ensure_ascii=False)
    opts = "".join(f'<label><input type="checkbox" value="{k}"> {html.escape(v)}</label>' for k, v in TOPICS)
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Topic audit {who}</title><style>
:root{{--bg:#fafaf8;--ink:#222;--sec:#666;--card:#fff;--line:#ddd;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1b1b1a;--ink:#eee;--sec:#aaa;--card:#262625;--line:#444;--acc:#6ea4e6}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:760px;padding:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}}
.post{{white-space:pre-wrap;font-size:16px;margin-bottom:12px}} label{{display:inline-block;margin:3px 10px 3px 0}}
button{{font:inherit;padding:6px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}}
.acc{{background:var(--acc);color:#fff;border-color:var(--acc)}} .sec{{color:var(--sec);font-size:13px}}
</style></head><body>
<h2>Topic audit, annotator {who}</h2>
<p class="sec">For each post: is it about cybersecurity? If yes, tick every topic that applies. Use "unsure" when the text
does not let you decide. No system labels are shown. Progress is saved in this browser; export the CSV when done.</p>
<div id="nav"></div><div id="item"></div>
<p><button onclick="go(-1)">Previous</button> <button class="acc" onclick="go(1)">Next</button>
<button onclick="exportCsv()">Export CSV</button> <span id="prog" class="sec"></span></p>
<script>
const D={data};const KEY="topic_audit_{who}";let S={{}};try{{S=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{}}
let i=0;try{{i=+localStorage.getItem(KEY+"_i")||0}}catch(e){{}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(S));localStorage.setItem(KEY+"_i",i)}}catch(e){{}}}}
function render(){{const d=D[i],s=S[d.id]||{{}};
document.getElementById("item").innerHTML=`<div class="card"><div class="sec">${{d.id}} · ${{i+1}} / ${{D.length}}</div>
<div class="post"></div><div><b>Cybersecurity?</b> <label><input type="radio" name="c" value="yes"> yes</label>
<label><input type="radio" name="c" value="no"> no</label><label><input type="radio" name="c" value="unsure"> unsure</label></div>
<div id="tp"><b>Topics</b><br>{opts}<label><input type="checkbox" value="topic_unsure"> unsure</label></div></div>`;
document.querySelector(".post").textContent=d.t;
document.querySelectorAll('input[name=c]').forEach(x=>{{x.checked=s.c===x.value;x.onchange=()=>{{(S[d.id]=S[d.id]||{{}}).c=x.value;save();prog()}}}});
document.querySelectorAll('#tp input').forEach(x=>{{x.checked=(s.t||[]).includes(x.value);x.onchange=()=>{{const t=[...document.querySelectorAll('#tp input:checked')].map(y=>y.value);(S[d.id]=S[d.id]||{{}}).t=t;save()}}}});prog()}}
function prog(){{document.getElementById("prog").textContent=Object.values(S).filter(x=>x.c).length+" of "+D.length+" done"}}
function go(k){{i=Math.max(0,Math.min(D.length-1,i+k));save();render()}}
function exportCsv(){{const rows=[["item_id","cyber","topics"]].concat(D.map(d=>[d.id,(S[d.id]||{{}}).c||"",((S[d.id]||{{}}).t||[]).join(";")]));
const b=new Blob([rows.map(r=>r.map(x=>'"'+String(x).replace(/"/g,'""')+'"').join(",")).join("\\n")],{{type:"text/csv"}});
const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="topic_audit_{who}.csv";a.click()}}
document.addEventListener("keydown",e=>{{if(e.key==="ArrowRight")go(1);if(e.key==="ArrowLeft")go(-1)}});render();
</script></body></html>""")


if __name__ == "__main__":
    main()
