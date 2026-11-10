"""Build the anonymous supplementary material for paper 1 (2026-11-09) → outputs/www_supp/supplementary.zip.

Contents: reproduce/ (paper data + table/figure generators; regenerates every number and figure), data/ (de-identified
main-sample export, audit labels without text), code/ (analysis scripts and the interaction collector, project name
replaced), preregistration/ (paper-1 registrations and deviations). Identifying strings are replaced and the result is
scanned; the build fails if any listed pattern remains.
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HOME = Path.home()
CC = HOME / "jingrui"
PAPER = HOME / "ccint-www2027-paper"
W27 = CC / "www2027"
OUT = CC / "outputs/www_supp"
ST = OUT / "supplementary"
BAD = re.compile(r"xinyu|w4ng77|astronautrron|pranalab|jingrui|ccint|@gmail|/home/|CCINT", re.I)
SUB = [(re.compile(r"CCINT_EXPORT_SALT"), "EXPORT_SALT"), (re.compile(r"\bccint\b"), "sensor"),
       (re.compile(r"ccint", re.I), "sensor"), (re.compile(r"/home/[^\s\"']+"), "<path>")]
SCRIPTS = ["www_gonogo.py", "www_eval_monitor.py", "www_eval_registry.py", "www_eval_contrast.py", "www_eval_sampling.py",
           "www_experiments.py", "www_producer.py", "www_sensitivity.py", "www_scale.py", "www_story_checks.py",
           "www_case_data.py", "www_teaser_data.py", "www_claim_timing.py", "www_crossover_fresh.py", "www_registry.py",
           "www_audit_score.py", "www_topic_audit.py", "www_annotation_tasks.py", "www_llm_annotator.py",
           "export_www_data.py", "interaction_descriptive.py", "interaction_collect_scale.py"]
FIGS = ["fig_cartoon.py", "fig_teaser.py", "fig_feedcases.py", "fig_crossover.py", "fig_eval.py"]


def scrub(text):
    for pat, rep in SUB:
        text = pat.sub(rep, text)
    return text


def put(src, dst, text=True):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if text:
        dst.write_text(scrub(src.read_text(encoding="utf-8")), encoding="utf-8")
    else:
        shutil.copy(src, dst)


def main():
    if ST.exists():
        shutil.rmtree(ST)
    # reproduce/: data + generators, laid out as in the paper repository
    for f in sorted((PAPER / "data").glob("*.json")):
        put(f, ST / "reproduce/data" / f.name)
    for f in ("gen_tables.py", "gen_audit.py"):
        put(PAPER / "tables" / f, ST / "reproduce/tables" / f)
    for f in FIGS:
        put(PAPER / "figures" / f, ST / "reproduce/figures" / f)
    # data/: de-identified main-sample export (hashed accounts; public handles only for news/organisation/feed accounts)
    for f in ("seeds_with_feed_flags.csv", "seeds_with_feed_flags.meta.json", "edges.csv"):
        put(W27 / "data/www_export" / f, ST / "data/main_sample" / f)
    # audit labels without post text: keys (item, post id, stratum, system label) and labels (author, model rater)
    A = CC / "outputs/www_audit"
    for t, keyf, lab in (("T1_topic", "topic/key.csv", "topic_audit"), ("T2_extraction", "extraction/key.csv", "extraction"),
                         ("T3_matching", "matching/key.csv", "matching"), ("T4_domain", "domain/key.csv", "domain")):
        put(A / keyf, ST / f"data/audit/{t}_key.csv")
        put(A / f"returned/{lab}_B.csv", ST / f"data/audit/{t}_labels_author.csv")
        put(A / f"returned/llm_A/{lab}_A_llm.csv", ST / f"data/audit/{t}_labels_model.csv")
    # code/
    for f in SCRIPTS:
        put(CC / "scripts" / f, ST / "code/scripts" / f)
    for f in sorted((CC / "src/ccint/interaction").glob("*.py")):
        put(f, ST / "code/sensor/interaction" / f.name)
    # preregistration/ (paper-1 parts only)
    for f in ("PREREG.md", "PREREG_CROSSOVER.md", "PREREG_CROSSOVER_AMEND1.md"):
        put(W27 / "results/frozen_eval" / f, ST / "preregistration" / f)
    dev = (W27 / "results/frozen_eval/DEVIATIONS.md").read_text()
    dev = dev.split("## Paper 2")[0]
    (ST / "preregistration/DEVIATIONS.md").write_text(scrub(dev))
    (ST / "README.md").write_text(README)
    # verify: regenerate the number macros and compare with the paper's
    tmp = ST / "reproduce/tables"
    (tmp / "numbers.tex").unlink(missing_ok=True)
    subprocess.run([sys.executable, "tables/gen_tables.py"], cwd=ST / "reproduce", check=True, capture_output=True)
    subprocess.run([sys.executable, "tables/gen_audit.py"], cwd=ST / "reproduce", check=True, capture_output=True)
    same = (tmp / "numbers.tex").read_text() == (PAPER / "tables/numbers.tex").read_text() and \
           (tmp / "audit.tex").read_text() == (PAPER / "tables/audit.tex").read_text()
    for f in tmp.glob("*.tex"):
        f.unlink()
    # scan
    hits = []
    for f in ST.rglob("*"):
        if f.is_file():
            for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                if BAD.search(line):
                    hits.append(f"{f.relative_to(ST)}:{i}: {line[:100]}")
    if hits:
        print("\n".join(hits[:40]))
        sys.exit(f"identifying strings remain: {len(hits)}")
    z = OUT / "supplementary.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(ST.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(OUT))
    print("numbers reproduced:", same)
    print(z, round(z.stat().st_size / 1e6, 2), "MB")
    for f in sorted(ST.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(ST)}  {f.stat().st_size // 1024} KB")


README = """# Supplementary material

Anonymous supplementary material for the paper "What Gets Counted Gets Found: The Diverging Signals of Events and
Engagement". Posts are public Bluesky posts; this package contains no post text, no URIs and no account
identifiers (accounts are salted hashes; a public handle is kept only for accounts profiled as news, organisation or
security-feed accounts).

## reproduce/
Every number and figure in the paper is generated from the JSON files in `reproduce/data/`:

    cd reproduce
    python tables/gen_tables.py      # number macros and tables (tables/*.tex)
    python tables/gen_audit.py       # audit macros (tables/audit.tex)
    python figures/fig_eval.py       # and the other fig_*.py: figures/*.pdf

Requirements: Python 3.10+, numpy, scipy, matplotlib (figures use pgf with a LaTeX installation).

## data/
- `main_sample/`: seed posts of the main sample (topic, actor type, human feed labels, outcome flags) and their reply
  and quote edges (hashed source account, depth, time, delay).
- `audit/`: the human audits T1-T4 (one author) and the model rater (Qwen3.5-4B), as item id, internal post id,
  sampling stratum and weight, the system's label, and the raters' labels. Post text is omitted.

## code/
Analysis scripts that produced the JSON files (`code/scripts/`) and the interaction collector (`code/sensor/`). They
read a PostgreSQL database built by the collector from the public Bluesky API and are included for inspection.

## preregistration/
Pre-registrations written before the corresponding data were collected or evaluated, the amendment, and the log of
deviations.
"""

if __name__ == "__main__":
    main()
