"""
Compare two saved evaluation runs.

    python -m evals.compare BASELINE.json CANDIDATE.json
    python -m evals.compare BASELINE.json CANDIDATE.json --fail-on-regression

Shows which run components changed (prompt, tool schemas, model, dataset,
policy index, retrieval settings, cases), every metric with its change
(better / worse by the metric's direction), and cases whose result
changed. With --fail-on-regression the exit code is 1 when a gated
metric got worse or a case went from pass to fail.
"""

import argparse
import json
import sys
from pathlib import Path

from evals.scorers import GATED, HIGHER_IS_BETTER, LOWER_IS_BETTER


def load(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("run", "metrics", "cases"):
        if key not in report:
            raise ValueError(f"{path}: not an evaluation report (missing {key!r})")
    return report


def compare(baseline, candidate):
    """Structured comparison of two reports (pure; used by the CLI and tests)."""

    base_c, cand_c = baseline["run"].get("components", {}), candidate["run"].get("components", {})
    changed_components = sorted(k for k in set(base_c) | set(cand_c) if base_c.get(k) != cand_c.get(k))

    metrics = []
    for name in list(dict.fromkeys(list(baseline["metrics"]) + list(candidate["metrics"]))):
        a, b = baseline["metrics"].get(name), candidate["metrics"].get(name)
        status = "same"
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a != b:
            if name in HIGHER_IS_BETTER:
                status = "better" if b > a else "worse"
            elif name in LOWER_IS_BETTER:
                status = "better" if b < a else "worse"
            else:
                status = "changed"
        elif a != b:
            status = "changed"
        metrics.append({"metric": name, "baseline": a, "candidate": b, "status": status,
                        "gated": name in GATED,
                        "delta": round(b - a, 4) if isinstance(a, (int, float)) and isinstance(b, (int, float))
                        else None})

    base_cases = {c["id"]: c["passed"] for c in baseline["cases"]}
    cand_cases = {c["id"]: c["passed"] for c in candidate["cases"]}
    case_changes = []
    for case_id in sorted(set(base_cases) | set(cand_cases)):
        a, b = base_cases.get(case_id), cand_cases.get(case_id)
        if a != b:
            kind = ("added" if a is None else "removed" if b is None
                    else "fixed" if b else "regressed")
            case_changes.append({"id": case_id, "baseline": a, "candidate": b, "change": kind})

    regressions = [m["metric"] for m in metrics if m["gated"] and m["status"] == "worse"]
    regressions += [f"case:{c['id']}" for c in case_changes if c["change"] == "regressed"]
    return {"changed_components": changed_components, "metrics": metrics,
            "case_changes": case_changes, "regressions": regressions,
            "same_cases": baseline["run"].get("cases", {}).get("sha256") == candidate["run"].get("cases", {}).get("sha256"),
            "modes": (baseline["run"].get("mode"), candidate["run"].get("mode"))}


def _fmt(value):
    return "-" if value is None else (f"{value:g}" if isinstance(value, float) else str(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare two BankMate evaluation reports.")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="Exit 1 if a gated metric got worse or a case regressed")
    args = parser.parse_args(argv)

    result = compare(load(args.baseline), load(args.candidate))
    print(f"\nBASELINE  {args.baseline}\nCANDIDATE {args.candidate}")
    if result["modes"][0] != result["modes"][1]:
        print(f"note: comparing a {result['modes'][0]} run with a {result['modes'][1]} run")
    if not result["same_cases"]:
        print("note: the case files differ; per-case changes include added/removed cases")
    print("\nChanged components: " + (", ".join(result["changed_components"]) or "none"))
    print(f"\n  {'metric':<26} {'baseline':>11} {'candidate':>11} {'delta':>9}  status")
    for m in result["metrics"]:
        flag = {"better": "better", "worse": "WORSE" if m["gated"] else "worse"}.get(m["status"], "")
        print(f"  {m['metric']:<26} {_fmt(m['baseline']):>11} {_fmt(m['candidate']):>11} "
              f"{_fmt(m['delta']):>9}  {flag}")
    if result["case_changes"]:
        print("\nCase changes")
        for c in result["case_changes"]:
            print(f"  {c['id']:<40} {c['change']}")
    print("\nRegressions: " + (", ".join(result["regressions"]) or "none"))
    return 1 if args.fail_on_regression and result["regressions"] else 0


if __name__ == "__main__":
    sys.exit(main())
