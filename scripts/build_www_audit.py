"""Build the two manual audits for the WWW short paper (offline HTML, no system labels shown).

    python scripts/build_www_audit.py
    → outputs/www_audit/{canada,feed}/annotate_{A,B}.html, items.csv, key.csv, meta.json

1. Canada relevance: simple random sample of 150 observed llm_v2d main seeds (rng 20261105).
   Question: is this a Canada-related cybersecurity post?  yes / no / unclear.
2. Feed status (account level): every account flagged by behavioral_v1 or volume_only, every unflagged
   account with >= 3 main seeds, and 40 random other accounts. Each card shows the handle, profile link,
   activity counts and up to 5 of the account's posts in the corpus. Labels: role (6 classes, as in
   ANNOTATION_GUIDELINES.md) and feed / non-feed / unclear.
A labels every item; B labels a reliability subset (Canada 60, feed 25 per stratum), independently; key.csv (system flags) is never shown.
Output contains post text and handles: it stays out of the public ccint repo (outputs/ is gitignored).
"""
from __future__ import annotations

import csv
import html
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location("desc", "scripts/interaction_descriptive.py")
desc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(desc)
from ccint import db  # noqa: E402
from ccint.analytics.broadcast import SQL as BSQL, detect_broadcast_authors  # noqa: E402
from ccint.interaction.collect import EP_QUOTES, EP_THREAD  # noqa: E402

RNG = 20261105
OUT = Path("outputs/www_audit")
OK = {"success", "partial"}

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:20px auto;padding:0 16px;color:#222;background:#fff}
.card{border:1px solid #ddd;border-radius:6px;padding:12px;margin:8px 0}
.post{background:#f6f6f6;border-left:4px solid #bbb;padding:8px 10px;margin:6px 0;white-space:pre-wrap}
.meta{font-size:.85em;color:#555}
fieldset{border:1px solid #ddd;margin:10px 0;padding:6px 10px}
label{display:inline-block;margin:3px 12px 3px 0;cursor:pointer}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0}
button{padding:6px 12px} textarea{width:100%;height:40px} .done{color:green} .todo{color:#c00}
</style></head><body>
<h2>__TITLE__</h2><div class="meta">__INSTR__</div>
<div class="bar"><button id="prev">&larr; Prev</button><button id="next">Next &rarr;</button>
<span id="pos"></span><span id="stat"></span><button id="exp">Export CSV</button></div>
<div id="item"></div>
<script>
const ITEMS=__ITEMS__, Q=__QUESTIONS__, KEY="__STORE__";
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0; try{i=+localStorage.getItem(KEY+":i")||0}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(st));localStorage.setItem(KEY+":i",i)}catch(e){}}
function done(id){return Q.every(q=>st[id]&&st[id][q.k])}
function render(){
  const it=ITEMS[i], a=st[it.id]||{};
  let h='<div class="card">'+it.html;
  for(const q of Q){h+='<fieldset><legend>'+q.t+'</legend>';
    q.o.forEach((o,j)=>{h+='<label><input type="radio" name="'+q.k+'" value="'+o[0]+'"'+(a[q.k]===o[0]?' checked':'')+'> '+o[1]+'</label>'});
    h+='</fieldset>'}
  h+='<textarea id="note" placeholder="notes (optional)">'+(a.note||'')+'</textarea></div>';
  document.getElementById('item').innerHTML=h;
  document.querySelectorAll('input[type=radio]').forEach(r=>r.onchange=()=>{st[it.id]=st[it.id]||{};st[it.id][r.name]=r.value;save();status()});
  document.getElementById('note').oninput=e=>{st[it.id]=st[it.id]||{};st[it.id].note=e.target.value;save()};
  document.getElementById('pos').textContent=(i+1)+' / '+ITEMS.length+' ('+it.id+')';status()}
function status(){const n=ITEMS.filter(x=>done(x.id)).length;const s=document.getElementById('stat');
  s.textContent=n+' done'; s.className=n===ITEMS.length?'done':'todo'}
document.getElementById('prev').onclick=()=>{if(i>0){i--;save();render()}};
document.getElementById('next').onclick=()=>{if(i<ITEMS.length-1){i++;save();render()}};
document.getElementById('exp').onclick=()=>{const cols=['item_id',...Q.map(q=>q.k),'note'];
  const esc=v=>'"'+String(v??'').replace(/"/g,'""')+'"';
  const rows=[cols.join(',')].concat(ITEMS.map(x=>{const a=st[x.id]||{};return [x.id,...Q.map(q=>a[q.k]||''),a.note||''].map(esc).join(',')}));
  const b=new Blob([rows.join('\n')],{type:'text/csv'});const u=URL.createObjectURL(b);
  const l=document.createElement('a');l.href=u;l.download=KEY+'.csv';l.click()};
render();
</script></body></html>"""


def page(title, instr, items, questions, store):
    return (PAGE.replace("__TITLE__", html.escape(title)).replace("__INSTR__", instr)
            .replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/")).replace("__QUESTIONS__", json.dumps(questions))
            .replace("__STORE__", store))


def url(u, handle):
    return html.escape(u or "")


def write(dirn, title, instr, items, questions, key_rows, meta):
    d = OUT / dirn
    d.mkdir(parents=True, exist_ok=True)
    # A labels every item (primary labels). B labels a reliability subset only (decided 2026-11-05):
    # Canada 60 random items; feed 25 per stratum. Separate RNG so A's page and item ids never change.
    sub = random.Random(RNG + 1)
    if dirn == "canada":
        keep = set(sub.sample([it["id"] for it in items], 60))
    else:
        keep = set()
        for strat in ("flagged", "unflagged_3plus", "random_rest"):
            keep |= set(sub.sample([k["item_id"] for k in key_rows if k["stratum"] == strat], 25))
    b_items = [it for it in items if it["id"] in keep]
    for who, its in (("A", items), ("B", b_items)):
        (d / f"annotate_{who}.html").write_text(page(f"{title} - annotator {who}", instr, its, questions,
                                                     f"www_{dirn}_{who}"))
    meta = {**meta, "B_subset": sorted(keep), "B_n": len(keep)}
    with open(d / "items.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "text"])
        w.writerows([[it["id"], it["plain"]] for it in items])
    with open(d / "key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        w.writeheader()
        w.writerows(key_rows)
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))


def main() -> None:
    m, runs, seeds, oc, edges, info, tc = desc.load()

    def observed(s):
        v = oc.get(s)
        return bool(v) and v[EP_THREAD]["outcome"] in OK and v[EP_QUOTES]["outcome"] in OK
    main_obs = sorted(s for s, r in seeds.items() if r["in_main"] == "true" and observed(s))
    with db.connect() as c:
        beh = detect_broadcast_authors(c, label_version="llm_v2d").author_ids
        agg = {r["author_id"]: r for r in c.execute(BSQL, {"label_version": "llm_v2d", "since": None, "until": None,
                                                         "min_lag_hours": 24})}
        vol = {a for a, r in agg.items() if r["n_posts"] >= 5 and r["n_days"] >= 4}
        text = {r["post_id"]: (r["text"], r["url"]) for r in c.execute(
            "SELECT post_id, text, url FROM posts WHERE post_id = ANY(%s)", (main_obs,))}
    rng = random.Random(RNG)

    # 1. Canada relevance
    samp = sorted(rng.sample(main_obs, 150))
    items, key = [], []
    for n, s in enumerate(samp, 1):
        t, uri = text[s]
        h = info[s]["author_handle"]
        iid = f"C{n:03d}"
        items.append({"id": iid, "plain": t, "html": f'<div class="post">{html.escape(t or "")}</div>'
                      f'<div class="meta"><a href="{url(uri, h)}" target="_blank">open on Bluesky</a> · '
                      f'{info[s]["published_at"]:%Y-%m-%d}</div>'})
        key.append({"item_id": iid, "post_id": s, "topic_v2": info[s]["topic"],
                    "feed_behavioral_v1": int(info[s]["author_id"] in beh), "feed_volume_only": int(info[s]["author_id"] in vol)})
    write("canada", "Canada relevance audit",
          "For each post: is it a <b>cybersecurity</b> post that is <b>related to Canada</b> (a Canadian victim, "
          "organisation, agency, law, location or audience)? A generic global story that merely mentions Canada in a "
          "list counts as yes only if Canada is part of what the post reports. Use <i>unclear</i> sparingly.",
          items, [{"k": "canada_cyber", "t": "Canada-related cybersecurity post?",
                   "o": [["yes", "yes"], ["no", "no"], ["unclear", "unclear"]]}],
          key, {"rng_seed": RNG, "population": "observed llm_v2d main seeds", "N": len(main_obs), "n": 150})

    # 2. Feed status, account level
    by_author = defaultdict(list)
    for s in main_obs:
        by_author[info[s]["author_id"]].append(s)
    flagged = sorted(a for a in by_author if a in beh or a in vol)
    hi = sorted(a for a in by_author if a not in beh | vol and len(by_author[a]) >= 3)
    rest = sorted(a for a in by_author if a not in beh | vol and len(by_author[a]) < 3)
    rnd = sorted(rng.sample(rest, 40))
    accts = flagged + hi + rnd
    rng.shuffle(accts)
    items, key = [], []
    for n, a in enumerate(accts, 1):
        ss = sorted(by_author[a], key=lambda s: info[s]["published_at"], reverse=True)
        h = info[ss[0]]["author_handle"]
        r = agg.get(a, {})
        posts = "".join(f'<div class="post">{html.escape(text[s][0] or "")}<div class="meta">'
                        f'{info[s]["published_at"]:%Y-%m-%d %H:%M} · <a href="{url(text[s][1], h)}" target="_blank">open</a>'
                        f'</div></div>' for s in ss[:5])
        iid = f"F{n:03d}"
        items.append({"id": iid, "plain": h, "html":
                      f'<div><b>@{html.escape(h)}</b> · <a href="https://bsky.app/profile/{html.escape(h)}" target="_blank">'
                      f'profile</a></div><div class="meta">{r.get("n_posts", "?")} relevant posts on {r.get("n_days", "?")} '
                      f'days in the corpus; showing the latest {min(5, len(ss))} of its {len(ss)} seed posts</div>{posts}'})
        key.append({"item_id": iid, "author_id": a, "handle": h, "n_main_seeds": len(ss),
                    "stratum": "flagged" if a in flagged else "unflagged_3plus" if a in hi else "random_rest",
                    "feed_behavioral_v1": int(a in beh), "feed_volume_only": int(a in vol),
                    "actor_type_v1": info[ss[0]]["actor_type"]})
    write("feed", "Feed account audit",
          "Judge the <b>account</b>, using its posts and (if needed) its profile page. "
          "<b>Feed</b> = posts are generated automatically or from a template, at high rate, mirroring a database, "
          "leak site, advisory list or RSS source, with no sign of a person writing them. Official organisation and "
          "news accounts that write their own posts are <i>not</i> feeds, even if they post often. "
          "Roles follow ANNOTATION_GUIDELINES.md.",
          items, [{"k": "role", "t": "Account role", "o": [
                      ["automated_feed", "automated feed"], ["organizational_broadcast", "organisation (official)"],
                      ["news_media", "news / media"], ["security_professional", "security professional"],
                      ["individual", "individual"], ["unclear", "unclear"]]},
                  {"k": "feed", "t": "Is this a feed account?", "o": [["feed", "feed"], ["non_feed", "non-feed"],
                                                                     ["unclear", "unclear"]]}],
          key, {"rng_seed": RNG, "strata": dict(Counter(k["stratum"] for k in key)), "n": len(key),
                "population_authors": len(by_author)})
    print(json.dumps({"canada": 150, "feed": dict(Counter(k["stratum"] for k in key))}))


if __name__ == "__main__":
    main()
