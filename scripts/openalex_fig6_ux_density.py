"""Figure 6: share of UX-framed records within each Human-AI theme, 2022 vs 2025.

Data: OpenAlex works API (Priem et al., 2022), title_and_abstract.search.
Themes C1-C5 follow the clusters of Gomez-Cruz et al. (2026, Sec. 3.2),
approximated by keyword proxies; each theme is crossed with a UX phrase set.

    python openalex_fig6_ux_density.py                 # query OpenAlex, then plot
    python openalex_fig6_ux_density.py --offline       # re-plot from saved counts

Requires: requests, pandas, matplotlib.
"""
import argparse
import datetime as dt
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

API = "https://api.openalex.org/works"
CUTOFF = "2026-11-06"  # publication-date cutoff used in the paper
BASE = ('("human-AI" OR "human-centered AI" OR "human-centric AI" '
        'OR "human-artificial intelligence")')
UX = ('("user experience" OR "UX" OR "user interface" OR "interaction design" '
      'OR "user-centered design" OR "human-centered design")')
THEMES = {
    "C1 Human–AI Interactions": '("explainable AI" OR XAI OR "human-in-the-loop" OR interpretability '
                                'OR transparency OR "trust calibration" OR "interactive machine learning")',
    "C2 Teaming and Augmentation": '("human-AI teaming" OR "human-agent teaming" OR "human-autonomy teaming" '
                                   'OR teamwork OR augmentation OR "human-AI cooperation" OR "decision-making")',
    "C3 Human–AI Collaboration": '("human-AI collaboration" OR "co-creation" OR co-creativity '
                                 'OR "hybrid intelligence" OR "collective intelligence" OR "human-AI symbiosis")',
    "C4 Conversational AI": '(chatbot OR "conversational agent" OR "conversational AI" '
                            'OR "natural language processing" OR empathy OR personalization)',
    "C5 Ethics and Responsibility": '("AI ethics" OR "responsible AI" OR trustworthiness OR privacy '
                                    'OR "participatory design" OR sustainability)',
}
YEARS = (2022, 2025)


def yearly_counts(query, raw_dir, tag):
    """Records per publication year matching `query` in titles/abstracts."""
    params = {"filter": f"title_and_abstract.search:{query},"
                        f"publication_year:{YEARS[0]}-{YEARS[1]},to_publication_date:{CUTOFF}",
              "group_by": "publication_year"}
    for attempt in range(8):  # back off on rate limiting (HTTP 429)
        r = requests.get(API, params=params, timeout=60)
        if r.status_code != 429:
            break
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    (raw_dir / f"{tag}.json").write_text(json.dumps(
        {"url": r.url, "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
         "response": r.json()}, indent=1))
    time.sleep(1)
    return {int(g["key"]): g["count"] for g in r.json()["group_by"]}


def fetch(out):
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, kw) in enumerate(THEMES.items(), start=1):
        theme = yearly_counts(f"{BASE} AND {kw}", raw, f"C{i}_theme")
        ux = yearly_counts(f"{BASE} AND {kw} AND {UX}", raw, f"C{i}_ux")
        rows += [{"theme": name, "year": y, "n_theme": theme.get(y, 0), "n_ux": ux.get(y, 0)}
                 for y in YEARS]
    df = pd.DataFrame(rows)
    df.to_csv(out / "fig6_counts.csv", index=False)
    return df


def plot(df, out):
    ink, ink2, grey, grey_l, accent = "#0b0b0b", "#52514e", "#8f8e88", "#cfcec8", "#2a78d6"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": grey_l,
                         "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    df = df.assign(pct=100 * df.n_ux / df.n_theme)
    w = df.pivot(index="theme", columns="year", values=["pct", "n_ux"])
    w = w.sort_values(("pct", YEARS[1]))
    top = w.index[-1]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.30, 0.14, 0.66, 0.68])
    for i, (theme, r) in enumerate(w.iterrows()):
        p_old, p_new = r[("pct", YEARS[0])], r[("pct", YEARS[1])]
        ax.barh(i, p_new, height=0.55, color=accent if theme == top else grey)
        ax.plot([p_old, p_old], [i - 0.36, i + 0.36], color=ink, lw=2)
        ax.text(max(p_old, p_new) + 0.3, i,
                f"{p_new:.1f}%  (n={int(r[('n_ux', YEARS[1])])} in {YEARS[1]}; "
                f"{YEARS[0]}: {p_old:.1f}%, n={int(r[('n_ux', YEARS[0])])})",
                va="center", fontsize=10, color=ink)
    ax.set_yticks(range(len(w)))
    ax.set_yticklabels(w.index)
    ax.set_xlim(0, 22)
    ax.set_xlabel(f"Share of each theme's records that are UX-framed (%)   "
                  f"bar = {YEARS[1]}, tick = {YEARS[0]}")
    fig.text(0.04, 0.95, f"{top.split(' ', 1)[1]} is the most UX-dense theme",
             fontsize=17, weight="bold", color=ink, va="top")
    fig.text(0.04, 0.885, f"UX-framed records as a share of each theme, {YEARS[1]} vs {YEARS[0]}",
             fontsize=12, color=ink2, va="top")
    fig.text(0.04, 0.025, f"Source: OpenAlex API, title_and_abstract.search, "
                          f"publication dates to {CUTOFF}.", fontsize=9, color=grey)
    for ext in ("png", "svg"):
        fig.savefig(out / f"fig6_ux_density_by_theme.{ext}", dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("fig6_output"))
    ap.add_argument("--offline", action="store_true", help="re-plot from <out>/fig6_counts.csv")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(a.out / "fig6_counts.csv") if a.offline else fetch(a.out)
    plot(data, a.out)
    print(data.assign(pct=(100 * data.n_ux / data.n_theme).round(2)).to_string(index=False))
