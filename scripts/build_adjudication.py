"""Fillable adjudication page for the A/B disagreements of the two manual audits.

    python scripts/build_adjudication.py  → outputs/www_audit/adjudicate.html

Shows each disagreement with the same content the annotators saw, plus both labels, and asks for the
final label (and, for feed items, the final role). Export gives adjudicated.csv rows for both tasks.
"""
import csv
import html
import json
import re
from pathlib import Path

OUT = Path("outputs/www_audit")
R = OUT / "returned"


def items_of(page):
    t = page.read_text()
    m = re.search(r"const ITEMS=(\[.*?\]), Q=", t, re.S)
    return {it["id"]: it for it in json.loads(m.group(1).replace("<\\/", "</"))}


def rows(p):
    return {r["item_id"]: r for r in csv.DictReader(open(p))}


CA, CB = rows(R / "canada/A.csv"), rows(R / "canada/B.csv")
FA, FB = rows(R / "feed/A.csv"), rows(R / "feed/B.csv")
KF = rows(OUT / "feed/key.csv")
ci, fi = items_of(OUT / "canada/annotate_A.html"), items_of(OUT / "feed/annotate_A.html")
ROLES = [["automated_feed", "automated feed"], ["organizational_broadcast", "organisation (official)"],
         ["news_media", "news / media"], ["security_professional", "security professional"],
         ["individual", "individual"], ["unclear", "unclear"]]
items = []
for i in sorted(CB):
    if CB[i]["canada_cyber"] and CA[i]["canada_cyber"] != CB[i]["canada_cyber"]:
        items.append({"id": i, "task": "canada", "html": ci[i]["html"],
                      "ab": f"A: <b>{CA[i]['canada_cyber']}</b> &nbsp;·&nbsp; B: <b>{CB[i]['canada_cyber']}</b>",
                      "q": [{"k": "final_label", "t": "Final: Canada-related cybersecurity post?",
                             "o": [["yes", "yes"], ["no", "no"], ["unclear", "unclear"]]}]})
for i in sorted((i for i in FB if FB[i]["feed"] and FA[i]["feed"] != FB[i]["feed"]),
                key=lambda i: -int(KF[i]["n_main_seeds"])):
    items.append({"id": i, "task": "feed", "html": fi[i]["html"],
                  "ab": f"A: <b>{html.escape(FA[i]['role'])} / {FA[i]['feed']}</b> &nbsp;·&nbsp; "
                        f"B: <b>{html.escape(FB[i]['role'])} / {FB[i]['feed']}</b> &nbsp;·&nbsp; {KF[i]['n_main_seeds']} seed posts",
                  "q": [{"k": "final_label", "t": "Final: is this a feed account?",
                         "o": [["feed", "feed"], ["non_feed", "non-feed"], ["unclear", "unclear"]]},
                        {"k": "final_role", "t": "Final account role", "o": ROLES}]})

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Audit adjudication</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:20px auto;padding:0 16px;color:#222;background:#fff}
.card{border:1px solid #ddd;border-radius:6px;padding:12px;margin:8px 0}
.post{background:#f6f6f6;border-left:4px solid #bbb;padding:8px 10px;margin:6px 0;white-space:pre-wrap}
.meta{font-size:.85em;color:#555} .ab{background:#fff8dc;border-left:4px solid #e0a800;padding:8px 10px;margin:8px 0}
.tag{display:inline-block;font-size:.8em;background:#eee;border-radius:3px;padding:1px 6px;margin-right:6px}
fieldset{border:1px solid #ddd;margin:10px 0;padding:6px 10px} label{display:inline-block;margin:3px 12px 3px 0;cursor:pointer}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0} button{padding:6px 12px}
textarea{width:100%;height:40px} .done{color:green} .todo{color:#c00}
</style></head><body>
<h2>Audit adjudication (A vs B disagreements)</h2>
<div class="meta">Decide together, using the same rules as the annotation. Both original labels are kept;
your final label is used in the analysis. When done, press <b>Export CSV</b> and send the file back.</div>
<div class="bar"><button id="prev">&larr; Prev</button><button id="next">Next &rarr;</button>
<span id="pos"></span><span id="stat"></span><button id="exp">Export CSV</button></div>
<div id="item"></div>
<script>
const ITEMS=__ITEMS__, KEY="www_adjudication";
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0; try{i=+localStorage.getItem(KEY+":i")||0}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(st));localStorage.setItem(KEY+":i",i)}catch(e){}}
function done(it){return it.q.every(q=>st[it.id]&&st[it.id][q.k])}
function render(){
  const it=ITEMS[i], a=st[it.id]||{};
  let h='<div class="card"><span class="tag">'+it.task+'</span><span class="tag">'+it.id+'</span>'+it.html+'<div class="ab">'+it.ab+'</div>';
  for(const q of it.q){h+='<fieldset><legend>'+q.t+'</legend>';
    q.o.forEach(o=>{h+='<label><input type="radio" name="'+q.k+'" value="'+o[0]+'"'+(a[q.k]===o[0]?' checked':'')+'> '+o[1]+'</label>'});
    h+='</fieldset>'}
  h+='<textarea id="note" placeholder="reason (optional)">'+(a.note||'')+'</textarea></div>';
  document.getElementById('item').innerHTML=h;
  document.querySelectorAll('input[type=radio]').forEach(r=>r.onchange=()=>{st[it.id]=st[it.id]||{};st[it.id][r.name]=r.value;save();status()});
  document.getElementById('note').oninput=e=>{st[it.id]=st[it.id]||{};st[it.id].note=e.target.value;save()};
  document.getElementById('pos').textContent=(i+1)+' / '+ITEMS.length;status()}
function status(){const n=ITEMS.filter(done).length;const s=document.getElementById('stat');
  s.textContent=n+' done'; s.className=n===ITEMS.length?'done':'todo'}
document.getElementById('prev').onclick=()=>{if(i>0){i--;save();render()}};
document.getElementById('next').onclick=()=>{if(i<ITEMS.length-1){i++;save();render()}};
document.getElementById('exp').onclick=()=>{const esc=v=>'"'+String(v??'').replace(/"/g,'""')+'"';
  const rows=['task,item_id,final_label,final_role,note'].concat(ITEMS.map(x=>{const a=st[x.id]||{};
    return [x.task,x.id,a.final_label||'',a.final_role||'',a.note||''].map(esc).join(',')}));
  const b=new Blob([rows.join('\n')],{type:'text/csv'});const u=URL.createObjectURL(b);
  const l=document.createElement('a');l.href=u;l.download='www_adjudication.csv';l.click()};
render();
</script></body></html>"""
(OUT / "adjudicate.html").write_text(PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/")))
print(len(items), "items:", sum(x["task"] == "canada" for x in items), "canada,", sum(x["task"] == "feed" for x in items), "feed")
