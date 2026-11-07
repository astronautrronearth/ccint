"""PPT-ready figures for the OpenAlex Human-AI topic trends + UX lens.

Inputs (frozen):  artifacts/openalex_trends/2026-11-06/openalex_counts.csv
                  artifacts/openalex_trends/ppt_2026-11-06/ux_counts.csv
Outputs:          artifacts/openalex_trends/ppt_2026-11-06/figures/*.png|svg
                  artifacts/openalex_trends/ppt_2026-11-06/metrics_ux.csv
Run: .venv-openalex/bin/python scripts/openalex_ppt_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1] / "artifacts/openalex_trends"
OUT = ROOT / "ppt_2026-11-06"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK, INK2, GREY, GREY_L, ACCENT = "#0b0b0b", "#52514e", "#8f8e88", "#cfcec8", "#2a78d6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": GREY_L,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
})
SOURCE = "Source: OpenAlex API (api.openalex.org/works), title_and_abstract.search, retrieved 2026-11-06."

counts = pd.read_csv(ROOT / "2026-11-06/openalex_counts.csv")
ux = pd.read_csv(OUT / "ux_counts.csv")
c = pd.concat([counts, ux]).pivot(index="category", columns="year", values="n").fillna(0)
BASE = "BASE corpus (human-AI)"
UXB = "UX (within human-AI base)"
THEMES = [k for k in c.index if k.startswith("C")]
SELECTED = {"C3 Human-AI Collaboration", "C4 Conversational AI", "C5 Ethics and Responsibility"}
short = lambda k: k.replace("Human-AI", "Human–AI")


def header(fig, title, subtitle):
    fig.text(0.04, 0.95, title, fontsize=17, weight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, subtitle, fontsize=12, color=INK2, va="top")
    fig.text(0.04, 0.025, SOURCE, fontsize=9, color=GREY)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def new(h=5.625):
    fig = plt.figure(figsize=(10, h))
    ax = fig.add_axes([0.08, 0.14, 0.88, 0.66])
    return fig, ax


# ---------- metrics table ----------
rows = []
for k in [BASE, UXB] + THEMES:
    rows.append({"category": k, "n_2021": c.loc[k, 2021], "n_2022": c.loc[k, 2022], "n_2025": c.loc[k, 2025],
                 "n_2026_partial": c.loc[k, 2026], "growth_2025_over_2022": c.loc[k, 2025] / c.loc[k, 2022],
                 "share_2021_pct": 100 * c.loc[k, 2021] / c.loc[BASE, 2021],
                 "share_2025_pct": 100 * c.loc[k, 2025] / c.loc[BASE, 2025],
                 "share_2026_partial_pct": 100 * c.loc[k, 2026] / c.loc[BASE, 2026]})
for t in THEMES:
    u = f"UX x {t}"
    rows.append({"category": u, "n_2021": c.loc[u, 2021], "n_2022": c.loc[u, 2022], "n_2025": c.loc[u, 2025],
                 "n_2026_partial": c.loc[u, 2026], "growth_2025_over_2022": c.loc[u, 2025] / c.loc[u, 2022],
                 "ux_share_of_theme_2022_pct": 100 * c.loc[u, 2022] / c.loc[t, 2022],
                 "ux_share_of_theme_2025_pct": 100 * c.loc[u, 2025] / c.loc[t, 2025]})
m = pd.DataFrame(rows).set_index("category")
m["share_change_pp"] = m.share_2025_pct - m.share_2021_pct
m.round(3).to_csv(OUT / "metrics_ux.csv")

# ---------- F1 base corpus growth ----------
fig, ax = new()
yrs = [2021, 2022, 2023, 2024, 2025, 2026]
vals = [c.loc[BASE, y] for y in yrs]
for y, v in zip(yrs, vals):
    partial = y == 2026
    ax.bar(y, v, width=0.62, color="white" if partial else GREY, edgecolor=GREY, hatch="///" if partial else None,
           linewidth=1)
    ax.text(y, v + 400, f"{int(v):,}" + ("\n(to 6 Nov)" if partial else ""), ha="center", va="bottom",
            fontsize=11, color=INK)
ax.bar(2025, c.loc[BASE, 2025], width=0.62, color=ACCENT)
ax.set_xticks(yrs)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_ylim(0, max(vals) * 1.22)
g = c.loc[BASE, 2025] / c.loc[BASE, 2021]
header(fig, f"Human–AI research grew {g:.0f}× in four years",
       f"OpenAlex records matching the Human–AI base query, by publication year (2021: {int(vals[0])} → 2025: {int(vals[4]):,})")
save(fig, "F1_base_corpus_growth")

# ---------- F2 two-metric selection ----------
fig, ax = new()
mt = m.loc[THEMES]
thr_g = mt.growth_2025_over_2022.sort_values(ascending=False).iloc[2]
thr_s = mt.share_change_pp.sort_values(ascending=False).iloc[2]
ax.axvspan(thr_g - 0.15, 23.5, ymin=0, ymax=1, color=ACCENT, alpha=0.0)
ax.fill_between([thr_g - 0.3, 23.5], thr_s - 0.6, 21.5, color=ACCENT, alpha=0.08, linewidth=0)
ax.text(thr_g - 0.1, 21, "Top 3 on both metrics", ha="left", va="top", fontsize=10.5, color=ACCENT)
ax.axhline(0, color=GREY_L, lw=1)
off = {"C1 Human-AI Interactions": (0.3, -1.6), "C2 Teaming and Augmentation": (0.3, 0.9),
       "C3 Human-AI Collaboration": (-0.3, -1.8), "C4 Conversational AI": (0.25, -1.6),
       "C5 Ethics and Responsibility": (-0.3, 1.0)}
for k, r in mt.iterrows():
    col = ACCENT if k in SELECTED else GREY
    ax.scatter(r.growth_2025_over_2022, r.share_change_pp, s=110, color=col, edgecolor="white", lw=2, zorder=3)
    dx, dy = off[k]
    ax.text(r.growth_2025_over_2022 + dx, r.share_change_pp + dy,
            f"{short(k)}\n{r.growth_2025_over_2022:.1f}×, {r.share_change_pp:+.1f} pp",
            ha="right" if dx < 0 else "left", va="center", fontsize=10, color=INK)
ax.set_xlim(8, 23.5)
ax.set_ylim(-5, 21.5)
ax.set_xlabel("Growth multiple, 2025 count ÷ 2022 count")
ax.set_ylabel("Share change 2021→2025 (pp)")
header(fig, "C3, C4 and C5 lead on both growth and share gain",
       "Five thematic keyword proxies inside the Human–AI base corpus; selection = top 3 on each metric, no weighting")
save(fig, "F2_theme_selection_scatter")

# ---------- F3 share dumbbell (themes + UX) ----------
fig, ax = new(6.0)
order = mt.sort_values("share_2025_pct").index.tolist()
order = [UXB] + order
for i, k in enumerate(order):
    r = m.loc[k]
    col = ACCENT if k == UXB else (INK2 if k in SELECTED else GREY)
    ax.plot([r.share_2021_pct, r.share_2025_pct], [i, i], color=col, lw=2.5, alpha=0.6)
    ax.scatter(r.share_2021_pct, i, s=60, facecolor="white", edgecolor=col, lw=2, zorder=3)
    ax.scatter(r.share_2025_pct, i, s=70, color=col, zorder=3)
    lo, hi = sorted([r.share_2021_pct, r.share_2025_pct])
    ax.text(hi + 0.8, i, f"{r.share_2021_pct:.1f}% → {r.share_2025_pct:.1f}%  ({r.share_change_pp:+.1f} pp)",
            va="center", fontsize=10.5, color=ACCENT if k == UXB else INK)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(["UX-framed work" if k == UXB else short(k) for k in order])
ax.get_yticklabels()[0].set_color(ACCENT)
ax.set_xlim(0, 52)
ax.set_xlabel("Share of Human–AI base corpus (%)   ○ 2021   ● 2025")
fig.axes[0].set_position([0.30, 0.14, 0.66, 0.68])
header(fig, "C3 doubled its share; UX-framed work slipped",
       "Share of the Human–AI base corpus per theme, 2021 vs 2025 (themes overlap, so shares do not sum to 100%)")
save(fig, "F3_share_shift_dumbbell")

# ---------- F4 indexed growth: UX vs field ----------
fig, ax = new()
ys = [2022, 2023, 2024, 2025]
for k in THEMES:
    ax.plot(ys, [c.loc[k, y] / c.loc[k, 2022] for y in ys], color=GREY_L, lw=1.5)
for k, col, lw, lab in [(BASE, INK2, 2.2, "All Human–AI"), (UXB, ACCENT, 3, "UX-framed")]:
    v = [c.loc[k, y] / c.loc[k, 2022] for y in ys]
    ax.plot(ys, v, color=col, lw=lw, marker="o", ms=6)
    ax.text(2025.08, v[-1], f"{lab} {v[-1]:.1f}×", va="center", color=col, fontsize=11, weight="bold")
ax.text(2025.08, c.loc["C3 Human-AI Collaboration", 2025] / c.loc["C3 Human-AI Collaboration", 2022],
        "Five themes\n(10.5×–21.7×)", va="center", color=GREY, fontsize=10)
ax.axhline(1, color=GREY_L, lw=1, ls=":")
ax.set_xticks(ys)
ax.set_xlim(2021.85, 2025.75)
ax.set_ylabel("Records, index 2022 = 1")
gu, gb = m.loc[UXB, "growth_2025_over_2022"], m.loc[BASE, "growth_2025_over_2022"]
header(fig, f"UX-framed work grew {gu:.1f}×, slower than the field ({gb:.1f}×)",
       "Growth of OpenAlex record counts, indexed to 2022; grey lines are the five thematic proxies")
save(fig, "F4_ux_vs_field_indexed")

# ---------- F5 UX share of base ----------
fig, ax = new()
s = [100 * c.loc[UXB, y] / c.loc[BASE, y] for y in yrs]
ax.plot(yrs[4:], s[4:], color=ACCENT, lw=2, ls="--", marker="o", ms=7, mfc="white")
ax.plot(yrs[:5], s[:5], color=ACCENT, lw=3, marker="o", ms=7)
for y, v in zip(yrs, s):
    ax.text(y, v + 0.35, f"{v:.1f}%", ha="center", fontsize=11, color=INK)
    ax.text(y, 0.4, f"n={int(c.loc[UXB, y]):,}", ha="center", fontsize=9.5, color=GREY)
ax.set_ylim(0, 8)
ax.set_xticks(yrs)
ax.set_xticklabels([str(y) for y in yrs[:5]] + ["2026\n(to 6 Nov)"])
ax.set_ylabel("UX-framed share of Human–AI records (%)")
header(fig, f"UX's share of Human–AI research peaked at {s[1]:.1f}% (2022), now {s[4]:.1f}%",
       "Human–AI records that also match a UX phrase, as % of all Human–AI records per year")
save(fig, "F5_ux_share_of_base")

# ---------- F6 UX density per theme ----------
fig, ax = new(6.0)
tk = sorted(THEMES, key=lambda t: m.loc[f"UX x {t}", "ux_share_of_theme_2025_pct"])
top = tk[-1]
for i, t in enumerate(tk):
    r = m.loc[f"UX x {t}"]
    ax.barh(i, r.ux_share_of_theme_2025_pct, height=0.55, color=ACCENT if t == top else GREY)
    ax.plot([r.ux_share_of_theme_2022_pct] * 2, [i - 0.36, i + 0.36], color=INK, lw=2)
    ax.text(max(r.ux_share_of_theme_2025_pct, r.ux_share_of_theme_2022_pct) + 0.3, i,
            f"{r.ux_share_of_theme_2025_pct:.1f}%  (n={int(r.n_2025)} in 2025; 2022: {r.ux_share_of_theme_2022_pct:.1f}%, n={int(r.n_2022)})",
            va="center", fontsize=10, color=INK)
ax.set_yticks(range(len(tk)))
ax.set_yticklabels([short(t) for t in tk])
ax.set_xlim(0, 22)
ax.set_xlabel("Share of each theme's records that are UX-framed (%)   bar = 2025, tick = 2022")
fig.axes[0].set_position([0.30, 0.14, 0.66, 0.68])
header(fig, f"{short(top).split(' ', 1)[1]} is the most UX-dense theme",
       "UX-framed records as a share of each theme, 2025 vs 2022 (2022 counts are small, ≤12 per theme)")
save(fig, "F6_ux_density_by_theme")

# ---------- F7 method pipeline ----------
fig = plt.figure(figsize=(10, 3.4))
ax = fig.add_axes([0, 0.1, 1, 0.72])
ax.set_xlim(0, 10)
ax.set_ylim(0.6, 2.4)
ax.axis("off")
steps = [("1  Base query", "4 Human–AI phrases\ntitle + abstract"),
         ("2  Theme proxies", "C1–C5 keyword sets\n+ UX phrase set"),
         ("3  Yearly counts", "OpenAlex group_by\n2021–2026\ncutoff 6 Nov 2026"),
         ("4  Two metrics", "growth 2025/2022\nshare change\n2021→2025"),
         ("5  Validate", "term-presence check\nmanual precision\n(pending)")]
w, gap = 1.7, 0.25
for i, (h, d) in enumerate(steps):
    x = 0.15 + i * (w + gap)
    col = ACCENT if i == 4 else GREY
    ax.add_patch(FancyBboxPatch((x, 0.75), w, 1.5, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=col, lw=1.8))
    ax.text(x + w / 2, 1.95, h, ha="center", va="center", fontsize=12, weight="bold", color=INK)
    ax.text(x + w / 2, 1.15, d, ha="center", va="center", fontsize=9.5, color=INK2)
    if i < 4:
        ax.annotate("", xy=(x + w + gap - 0.02, 1.5), xytext=(x + w + 0.02, 1.5),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.5))
fig.text(0.04, 0.95, "Method: keyword proxies over OpenAlex, ranked on two metrics", fontsize=17, weight="bold",
         color=INK, va="top")
fig.text(0.04, 0.04, SOURCE, fontsize=9, color=GREY)
save(fig, "F7_method_pipeline")
print(m.round(2).to_string())
