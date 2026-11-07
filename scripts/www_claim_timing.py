"""Post-hoc (2026-11-07, cold review): timing of registry claims relative to social posts, main window.
For every claim matched by any candidate cluster (exact): first candidate post vs registry publication; and for
claims retrieved by the posts ranking (b = 5), review day vs publication.
    python scripts/www_claim_timing.py → outputs/interaction_experiment/www_eval/claim_timing.json
"""
import importlib.util, json, sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
import numpy as np
sys.path[:0] = ["src", "scripts"]
from ccint import db
from ccint.interaction import manifest as mf


def _mod(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


REG = _mod("reg", "scripts/www_eval_registry.py")
m = mf.load("www_scale")
ids = [int(r["post_id"]) for r in m.population("in_scale")]
obs, edges, info = REG.SC.load(REG.SC.scale_runs(m.sha256), ids)
hist_ids = [s for s in ids if info.get(s) and info[s]["published_at"].date() in REG.HIST]
feeds = REG.SC.feed_rules(hist_ids, info)["volume_only"]
vic = [json.loads(l) for l in open("outputs/interaction_experiment/www_eval/victims.jsonl")]
with db.connect() as c:
    reg = c.execute("""SELECT event_id, title, published_at_utc FROM external_events WHERE source = 'ransomware.live'
                       AND published_at_utc >= '2026-08-08' AND published_at_utc < '2026-10-08'""").fetchall()
clusters = defaultdict(list)
for v in vic:
    if v["post_id"] in info and v["victims"]:
        for name in sorted({REG.norm(x) for x in v["victims"]}):
            if len(name) >= 3:
                clusters[name].append(v["post_id"])
first = {k: min(info[s]["published_at"] for s in L) for k, L in clusters.items()}
regn = [(r["event_id"], REG.norm(r["title"] or ""), r["published_at_utc"]) for r in reg]
byn = defaultdict(list)
for e in regn:
    byn[e[1]].append(e)
lead, firstfeed = [], []
for k, L in clusters.items():
    for e in byn.get(k, []):
        if first[k].date() - timedelta(14) <= e[2].date() <= first[k].date() + timedelta(14):
            lead.append((e[2] - first[k]).total_seconds() / 86400)
            s0 = min(L, key=lambda s: info[s]["published_at"])
            firstfeed.append(info[s0]["author_id"] in feeds)
lead = np.array(lead)
out = {"n_matched_claims": len(lead), "share_post_before_listing": float((lead > 0).mean()),
       "median_days_listing_after_first_post": float(np.median(lead)),
       "share_first_post_by_feed": float(np.mean(firstfeed))}
print(out)
Path("outputs/interaction_experiment/www_eval/claim_timing.json").write_text(json.dumps(out, indent=1))
