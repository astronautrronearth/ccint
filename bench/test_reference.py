"""Regression check: the harness reproduces the reference summary (primary variant: low victims, top placement)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ccbench as cb  # noqa: E402

data = sys.argv[1] if len(sys.argv) > 1 else "outputs/ccbench_v1"
S = cb.load(data)
got = cb.summarize(cb.evaluate(cb.reference_methods(), S), S)
ref = json.loads((Path(__file__).parent / "reference_top.json").read_text())["summary"]
assert got == ref, "summary differs from reference_top.json"
print("ok: reference summary reproduced")
