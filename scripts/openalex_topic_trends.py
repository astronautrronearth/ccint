"""
Research-trend chart by thematic cluster (OpenAlex publication counts per year).

Taxonomy (the 5 clusters) is taken from:
  Gomez-Cruz, Rodriguez Castro, Rey-Sarmiento, Zarate-Torres & Moncada Nino (2026).
  Mapping human-AI relationships: Intellectual structure and conceptual insights.
  Technologies, 14(2), 83.  https://doi.org/10.3390/technologies14020083

Method note
-----------
The paper built its corpus from Scopus with a "human-AI" base query, then found
the clusters by VOSviewer co-word analysis. We cannot reproduce VOSviewer's
cluster assignment, so each cluster is approximated by a KEYWORD PROXY built
from the node names the paper lists for that cluster (Section 3.2).
Categories overlap (one paper can match several) - they are not a partition.

Run:   pip install requests pandas matplotlib
       python scripts/openalex_topic_trends.py
Outputs: openalex_counts.csv, openalex_queries.json, precision_sample.csv,
         topic_trends.png, topic_trends.svg, topic_rankings.csv and raw responses.
Default output: artifacts/openalex_trends/<timestamp>/ (never overwrites a run).
Optional environment: OPENALEX_MAILTO, OPENALEX_API_KEY.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional; anonymous calls work. Never submit the supplied placeholder address.
MAILTO = os.environ.get("OPENALEX_MAILTO", "").strip()
Y0, Y1 = 2021, 2026
API = "https://api.openalex.org/works"

# Broader base-corpus proxy; not an exact reproduction of the Scopus query.
BASE = ('("human-AI" OR "human-centered AI" OR "human-centric AI" '
        'OR "human-artificial intelligence")')

CLUSTERS = {
    "C1 Human-AI Interactions":
        '("explainable AI" OR XAI OR "human-in-the-loop" OR interpretability '
        'OR transparency OR "trust calibration" OR "interactive machine learning")',
    "C2 Teaming and Augmentation":
        '("human-AI teaming" OR "human-agent teaming" OR "human-autonomy teaming" '
        'OR teamwork OR augmentation OR "human-AI cooperation" OR "decision-making")',
    "C3 Human-AI Collaboration":
        '("human-AI collaboration" OR "co-creation" OR co-creativity '
        'OR "hybrid intelligence" OR "collective intelligence" OR "human-AI symbiosis")',
    "C4 Conversational AI":
        '(chatbot OR "conversational agent" OR "conversational AI" '
        'OR "natural language processing" OR empathy OR personalization)',
    "C5 Ethics and Responsibility":
        '("AI ethics" OR "responsible AI" OR trustworthiness OR privacy '
        'OR "participatory design" OR sustainability)',
}

BASE_NAME = "BASE corpus (human-AI)"


def flt(query, as_of):
    # Do not silently fall back to default.search: it also searches full text
    # and keywords, changing the scope of the measurement.
    return (f"title_and_abstract.search:{query},publication_year:{Y0}-{Y1},"
            f"to_publication_date:{as_of.isoformat()}")


def abstract_text(inverted_index):
    positions = sorted((pos, word) for word, offsets in (inverted_index or {}).items()
                       for pos in offsets)
    return " ".join(word for _, word in positions)


def trend_metrics(wide):
    """Two unweighted metrics using only 2021–2025; no 2026 ranking."""
    records = []
    for name in CLUSTERS:
        n21, n22, n25 = (int(wide.loc[y, name]) for y in (2021, 2022, 2025))
        b21, b25 = (int(wide.loc[y, BASE_NAME]) for y in (2021, 2025))
        share21 = n21 / b21 if b21 else float("nan")
        share25 = n25 / b25 if b25 else float("nan")
        records.append({"category": name, "n_2022": n22, "n_2025": n25,
                        "growth_2025_over_2022": n25 / n22 if n22 else float("nan"),
                        "share_2021_pct": 100 * share21, "share_2025_pct": 100 * share25,
                        "share_change_pp": 100 * (share25 - share21)})
    metrics = pd.DataFrame(records)
    metrics["growth_rank"] = metrics.growth_2025_over_2022.rank(ascending=False, method="min")
    metrics["share_change_rank"] = metrics.share_change_pp.rank(ascending=False, method="min")
    metrics["top3_on_both"] = (metrics.growth_rank <= 3) & (metrics.share_change_rank <= 3)
    return metrics.sort_values(["growth_rank", "share_change_rank"])


def plot_trends(wide, out, as_of, retrieved):
    # Counts and shares together show both absolute growth and relative momentum.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    partial = as_of < datetime.date(Y1, 12, 31)
    for name in CLUSTERS:
        for ax, values in zip(axes, (wide[name], 100 * wide[name] / wide[BASE_NAME])):
            full = values.loc[:Y1 - 1] if partial else values
            line, = ax.plot(full.index, full.values, marker="o", linewidth=2, label=name)
            if partial:
                ax.plot([Y1 - 1, Y1], values.loc[[Y1 - 1, Y1]].values,
                        linestyle="--", marker="o", color=line.get_color(), linewidth=1.5)
    base = wide[BASE_NAME]
    axes[0].plot(base.loc[:Y1 - 1].index if partial else base.index,
                 base.loc[:Y1 - 1].values if partial else base.values,
                 color="black", linestyle=":", label="Base corpus")
    if partial:
        axes[0].plot([Y1 - 1, Y1], base.loc[[Y1 - 1, Y1]].values,
                     color="black", linestyle="--")
    axes[0].set(title="Publication counts", ylabel="Number of OpenAlex works")
    axes[1].set(title="Share of the base corpus", ylabel="Share (%)")
    for ax in axes:
        ax.set_xticks(wide.index)
        ax.set_xlabel(f"Year ({Y1} partial, dashed)" if partial else "Publication year")
        ax.grid(alpha=0.25)
    fig.suptitle("Research Trends in Human-AI Relationships, 2021–2026")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055),
               ncol=3, frameon=False, fontsize=9)
    fig.text(0.02, 0.015,
             "Clusters: Gómez-Cruz et al. (2026), Technologies, 14(2), 83. "
             f"Source: OpenAlex; retrieved {retrieved}; publication cutoff {as_of}.\n"
             "Title/abstract keyword proxies; overlapping categories; all work types. "
             "2026 excluded from ranking. Precision review pending.", fontsize=8)
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.savefig(out / "topic_trends.png", dpi=300)
    fig.savefig(out / "topic_trends.svg")
    plt.close(fig)


def main():
    now = datetime.datetime.now(ZoneInfo("America/Toronto"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/openalex_trends") / now.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--as-of", type=datetime.date.fromisoformat, default=now.date())
    parser.add_argument("--sample-size", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.sample_size <= 100:
        parser.error("--sample-size must be between 1 and 100")
    if not datetime.date(2026, 1, 1) <= args.as_of <= now.date():
        parser.error("--as-of must be in 2026 or later, and not in the future")
    # Never overwrite frozen counts or completed manual labels.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir()
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    if os.environ.get("OPENALEX_API_KEY"):
        session.headers["Authorization"] = f"Bearer {os.environ['OPENALEX_API_KEY']}"
    queries = {BASE_NAME: BASE, **{k: f"{BASE} AND {v}" for k, v in CLUSTERS.items()}}
    rows, samples, request_log = [], [], []

    def fetch(params, filename):
        # Persist the public request parameters, never credentials or email.
        public_params = dict(params)
        if MAILTO:
            params = {**params, "mailto": MAILTO}
        r = session.get(API, params=params, timeout=60)
        if not r.ok:
            raise RuntimeError(f"OpenAlex HTTP {r.status_code}: {r.text[:1500]}")
        payload = r.json()
        (raw_dir / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
        request_log.append({"file": f"raw/{filename}", "params": public_params,
                            "retrieved_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        return payload

    for i, (name, query) in enumerate(queries.items()):
        print(f"Retrieving {name} ...", flush=True)
        filters = {"filter": flt(query, args.as_of)}
        payload = fetch({**filters, "group_by": "publication_year"}, f"{i}_counts.json")
        counts = {int(g["key"]): int(g["count"]) for g in payload["group_by"]}
        if sum(counts.values()) != int(payload["meta"]["count"]):
            raise ValueError(f"Incomplete yearly aggregation: {name}")
        rows.extend({"category": name, "year": y, "n": counts.get(y, 0)}
                    for y in range(Y0, Y1 + 1))
        n = min(args.sample_size, sum(counts.values()))
        if not n:
            continue
        payload = fetch({**filters, "sample": n, "seed": 42, "per-page": n,
                         "select": "id,title,publication_year,publication_date,doi,"
                                   "abstract_inverted_index,primary_location,type"}, f"{i}_sample.json")
        works = payload["results"]
        if len(works) != n or len({w["id"] for w in works}) != n:
            raise ValueError(f"Incomplete or duplicate sample: {name}")
        for w in works:
            location = w.get("primary_location") or {}
            samples.append({"category": name, "id": w["id"], "year": w["publication_year"],
                            "title": w["title"], "doi": w.get("doi"),
                            "url": location.get("landing_page_url") or w["id"],
                            "type": w["type"],
                            "abstract": abstract_text(w.get("abstract_inverted_index")),
                            "relevant(0/1)": "", "review_notes": ""})
    df = pd.DataFrame(rows)
    wide = df.pivot(index="year", columns="category", values="n")
    if (wide[list(CLUSTERS)].gt(wide[BASE_NAME], axis=0)).any().any():
        raise ValueError("A cluster exceeds the base corpus; check Boolean query semantics")
    out = args.output_dir
    df.to_csv(out / "openalex_counts.csv", index=False)
    pd.DataFrame(samples).to_csv(out / "precision_sample.csv", index=False)
    metrics = trend_metrics(wide)
    metrics.to_csv(out / "topic_rankings.csv", index=False)
    plot_trends(wide, out, args.as_of, now.date())
    metadata = {
        "retrieved": now.date().isoformat(), "timezone": "America/Toronto",
        "source": "OpenAlex", "endpoint": API, "queries": queries,
        "date_cutoff": args.as_of.isoformat(), "year_range": [Y0, Y1],
        "search_scope": "title_and_abstract.search", "work_type_filter": None,
        "language_filter": None, "seed": 42, "sample_size_per_category": args.sample_size,
        "precision_status": "pending manual review; all labels intentionally blank",
        "ranking_years": [2021, 2022, 2025],
        "selection_rule": "Top 3 on BOTH growth_2025_over_2022 and share_change_pp; no weighted score",
        "snapshot_warning": "Date cutoff restricts publication dates, not database indexing dates. "
                            "Only the saved responses freeze the database snapshot.",
        "paper": {"doi": "10.3390/technologies14020083", "cluster_section": "3.2"},
        "limitations": ["Keyword proxies, not VOSviewer membership", "Categories overlap",
                        "OpenAlex differs from Scopus; all work types included",
                        "2026 is partial", "Publication volume is not industry impact",
                        "Pooled precision sample does not establish year-specific precision"],
        "versions": {"python": platform.python_version(), "requests": requests.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "requests": request_log,
    }
    (out / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    metadata["sha256"] = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "openalex_queries.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(wide.to_string())
    print("\nPROVISIONAL ranking (precision review pending):")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
