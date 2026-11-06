"""Annotation tasks T2 (victim extraction) and T3 (registry name matching) for the WWW evaluation.
Plan and guidelines: www2027/docs/annotation/ANNOTATION_PLAN.md. Contains post text: outputs only, not committed.

    python scripts/www_annotation_tasks.py → outputs/www_audit/extraction/, outputs/www_audit/matching/

T2: 150 ransomware-topic P_scale roots (100 with an extracted victim, 50 without), seed 20261106; 50 double-labelled.
T3: the fuzzy-only candidate/registry pairs plus 60 random exact-match pairs, shuffled so the match type is hidden;
    60 double-labelled. Annotators see names (and for T2 the post text) only, never the policy or match type.
"""
from __future__ import annotations

import csv
import html
import importlib.util
import json
import random
import sys
from pathlib import Path

sys.path[:0] = ["src", "scripts"]
from ccint import db  # noqa: E402

rnd = random.Random(20261106)
STYLE = """:root{--bg:#fafaf8;--ink:#222;--sec:#666;--card:#fff;--line:#ddd;--acc:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1a;--ink:#eee;--sec:#aaa;--card:#262625;--line:#444;--acc:#6ea4e6}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:760px;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}
.post{white-space:pre-wrap;font-size:16px;margin-bottom:12px} label{display:inline-block;margin:3px 12px 3px 0}
button{font:inherit;padding:6px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
.acc{background:var(--acc);color:#fff;border-color:var(--acc)} .sec{color:var(--sec);font-size:13px}
input[type=text]{font:inherit;width:100%;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}"""


def page(path, title, intro, items, fields):
    """items: [{id, html}], fields: [(key, label, kind, options)] with kind radio|text."""
    data = json.dumps(items, ensure_ascii=False)
    fdef = json.dumps(fields, ensure_ascii=False)
    key = path.stem
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>
<h2>{html.escape(title)}</h2><p class="sec">{intro}</p><div id="item"></div>
<p><button onclick="go(-1)">Previous</button> <button class="acc" onclick="go(1)">Next</button>
<button onclick="exportCsv()">Export CSV</button> <span id="prog" class="sec"></span></p>
<script>
const D={data};const F={fdef};const KEY="{key}";let S={{}};try{{S=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{}}
let i=0;try{{i=+localStorage.getItem(KEY+"_i")||0}}catch(e){{}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(S));localStorage.setItem(KEY+"_i",i)}}catch(e){{}}}}
function render(){{const d=D[i],s=S[d.id]||{{}};let h=`<div class="card"><div class="sec">${{d.id}} · ${{i+1}} / ${{D.length}}</div>${{d.html}}`;
for(const [k,l,kind,opts] of F){{h+=`<div><b>${{l}}</b><br>`;if(kind==="radio"){{for(const o of opts)h+=`<label><input type="radio" name="${{k}}" value="${{o}}"> ${{o}}</label>`}}
else h+=`<input type="text" name="${{k}}">`;h+=`</div>`}}h+=`</div>`;document.getElementById("item").innerHTML=h;
for(const [k,l,kind] of F){{document.querySelectorAll(`[name="${{k}}"]`).forEach(x=>{{if(kind==="radio")x.checked=s[k]===x.value;else x.value=s[k]||"";
x.oninput=x.onchange=()=>{{(S[d.id]=S[d.id]||{{}})[k]=x.value;save();prog()}}}})}};prog()}}
function prog(){{document.getElementById("prog").textContent=Object.values(S).filter(x=>x[F[0][0]]).length+" of "+D.length+" done"}}
function go(k){{i=Math.max(0,Math.min(D.length-1,i+k));save();render()}}
function exportCsv(){{const rows=[["item_id"].concat(F.map(f=>f[0]))].concat(D.map(d=>[d.id].concat(F.map(f=>(S[d.id]||{{}})[f[0]]||""))));
const b=new Blob([rows.map(r=>r.map(x=>'"'+String(x).replace(/"/g,'""')+'"').join(",")).join("\\n")],{{type:"text/csv"}});
const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=KEY+".csv";a.click()}}
document.addEventListener("keydown",e=>{{if(e.target.tagName==="INPUT"&&e.target.type==="text")return;if(e.key==="ArrowRight")go(1);if(e.key==="ArrowLeft")go(-1)}});render();
</script></body></html>""")


def t2():
    out = Path("outputs/www_audit/extraction")
    out.mkdir(parents=True, exist_ok=True)
    vic = [json.loads(l) for l in open("outputs/interaction_experiment/www_eval/victims.jsonl")]
    with_v = [v for v in vic if v["victims"]]
    without = [v for v in vic if v["victims"] == []]
    pick = rnd.sample(with_v, 100) + rnd.sample(without, 50)
    rnd.shuffle(pick)
    with db.connect() as c:
        text = {r["post_id"]: r["text"] for r in c.execute("SELECT post_id, text FROM posts WHERE post_id = ANY(%s)",
                                                           ([v["post_id"] for v in pick],))}
    items, key = [], []
    for i, v in enumerate(pick, 1):
        iid = f"E{i:03d}"
        names = "; ".join(v["victims"]) or "(none extracted)"
        items.append({"id": iid, "html": f'<div class="post">{html.escape(text[v["post_id"]] or "")}</div>'
                                          f'<div class="sec">Extracted victim name(s): <b>{html.escape(names)}</b></div>'})
        key.append([iid, v["post_id"], names])
    double = set(r[0] for r in rnd.sample(key, 50))
    fields = [["extraction", "Are the extracted names the organisations reported as victims in this post?",
               "radio", ["all correct", "partly correct", "wrong", "none extracted, correctly", "unsure"]],
              ["missed", "Victim organisation(s) named in the post but not extracted (leave empty if none)", "text", []]]
    intro = ("Read the post. Judge only what the text says: a victim is an organisation the post reports as hit by "
             "ransomware, extortion or a data-theft claim. See ANNOTATION_PLAN.md, task T2.")
    page(out / "extraction_A.html", "Victim extraction audit, annotator A", intro, items, fields)
    page(out / "extraction_B.html", "Victim extraction audit, annotator B", intro, [x for x in items if x["id"] in double], fields)
    with open(out / "key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "post_id", "extracted", "double"])
        w.writerows([r + [r[0] in double] for r in key])
    return len(items), len(double)


def t3():
    out = Path("outputs/www_audit/matching")
    out.mkdir(parents=True, exist_ok=True)
    fuzzy = list(csv.DictReader(open("outputs/interaction_experiment/www_eval/registry_fuzzy_review.csv")))
    spec = importlib.util.spec_from_file_location("reg", "scripts/www_eval_registry.py")
    reg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reg)
    with db.connect() as c:
        evs = c.execute("""SELECT event_id, title FROM external_events WHERE source = 'ransomware.live'
                           AND published_at_utc >= '2026-08-08' AND published_at_utc < '2026-10-08'""").fetchall()
    vic = [json.loads(l) for l in open("outputs/interaction_experiment/www_eval/victims.jsonl")]
    names = {reg.norm(x) for v in vic for x in (v["victims"] or []) if len(reg.norm(x)) >= 3}
    exact = [(e["title"], e["title"], e["event_id"]) for e in evs if reg.norm(e["title"] or "") in names]
    # show the raw extracted spelling for exact matches where available
    raw = {}
    for v in vic:
        for x in v["victims"] or []:
            raw.setdefault(reg.norm(x), x)
    exact = [(raw.get(reg.norm(t), t), t, eid) for t, _, eid in exact]
    pairs = [("fuzzy", r["candidate_name"], r["registry_name"], r["registry_event_id"]) for r in fuzzy] + \
            [("exact", a, b, c) for a, b, c in rnd.sample(exact, min(60, len(exact)))]
    rnd.shuffle(pairs)
    items, key = [], []
    for i, (typ, a, b, eid) in enumerate(pairs, 1):
        iid = f"M{i:03d}"
        items.append({"id": iid, "html": f'<div>Name in a post: <b>{html.escape(a)}</b></div>'
                                          f'<div>Name in the registry: <b>{html.escape(b)}</b></div>'})
        key.append([iid, typ, a, b, eid])
    double = set(r[0] for r in rnd.sample(key, 60))
    fields = [["same", "Do the two names refer to the same organisation?", "radio", ["yes", "no", "unsure"]]]
    intro = "Judge from the names only; you may search the web for the organisation. See ANNOTATION_PLAN.md, task T3."
    page(out / "matching_A.html", "Registry name matching, annotator A", intro, items, fields)
    page(out / "matching_B.html", "Registry name matching, annotator B", intro, [x for x in items if x["id"] in double], fields)
    with open(out / "key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "match_type", "candidate", "registry", "event_id", "double"])
        w.writerows([r + [r[0] in double] for r in key])
    return len(items), len(double)


if __name__ == "__main__":
    print("T2", t2())
    print("T3", t3())
