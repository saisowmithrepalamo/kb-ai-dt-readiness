"""Recompute published manuscript quantities from archived JSON outputs.

This script does not refit models. It checks the files the manuscript says
were produced by analysis.py and sensitivity.py.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results_original"
OUT = ROOT / "results_new"
OUT.mkdir(parents=True, exist_ok=True)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    denom = 1 + z2 / n
    margin = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)
    return (centre - margin) / denom, (centre + margin) / denom


def main() -> None:
    results = json.loads((SRC / "results.json").read_text())
    sensitive = json.loads((SRC / "sensitivity_results.json").read_text())
    checks = []

    def add(name: str, got, expected, tol=1e-9, note=""):
        ok = got is not None and abs(float(got) - float(expected)) <= tol
        checks.append({"check": name, "got": got, "expected": expected, "ok": bool(ok), "note": note})

    add("n", results["n"], 5000)
    add("n_train", results["n_train"], 3749)
    add("n_test", results["n_test"], 1251)
    add("breach_rate_rounded", round(results["breach_rate"], 3), 0.358, 0)
    add("hard_rate_percent", round(100 * results["hard_rate"], 1), 9.3, 0)
    q1 = results["figure_risk"][0]
    q5 = results["figure_risk"][4]
    add("q1_rate", q1["n_breach"] / q1["n"], 94 / 131)
    add("q5_rate", q5["n_breach"] / q5["n"], 34 / 132)
    lo, hi = wilson(94, 131)
    add("q1_wilson_lo", q1["observed_lo"], lo, 1e-6)
    add("q1_wilson_hi", q1["observed_hi"], hi, 1e-6)
    lo, hi = wilson(34, 132)
    add("q5_wilson_lo", q5["observed_lo"], lo, 1e-6)
    add("q5_wilson_hi", q5["observed_hi"], hi, 1e-6)
    add("quintile_n", sum(row["n"] for row in results["figure_risk"]), 656)
    add("quintile_breaches", sum(row["n_breach"] for row in results["figure_risk"]), 284)
    kcid = sensitive["seed23_full"]["kcid"]
    data = sensitive["seed23_full"]["data_logistic"]
    add("utility_diff_seed23", kcid["utility"] - data["utility"], 1.211, 0.01)
    add("launched_kcid", kcid["launched_breaches"], 170, 0)
    add("launched_data", data["launched_breaches"], 198, 0)
    rows = sensitive["seed_rows"]
    def signs(key):
        vals = [row[key] for row in rows]
        pos = sum(v > 1e-12 for v in vals)
        neg = sum(v < -1e-12 for v in vals)
        zero = len(vals) - pos - neg
        return pos, neg, zero, sum(vals) / len(vals)

    pos, neg, zero, mean = signs("dLB_kcid_eucap")
    add("capped_kcid_fewer", neg, 30, 0)
    add("capped_kcid_more", pos, 15, 0)
    add("capped_tie", zero, 5, 0)
    pos, neg, zero, mean = signs("dLB_kcid_hybrid")
    add("hybrid_kcid_fewer", neg, 30, 0)
    add("hybrid_kcid_more", pos, 16, 0)
    add("hybrid_tie", zero, 4, 0)
    pos, neg, zero, mean = signs("dU_kcid_data")
    add("utility_vs_opportunity_all", pos, 50, 0)
    pos, neg, zero, mean = signs("dU_kcid_rulesonly")
    add("utility_vs_rules_all", pos, 50, 0)
    pos, neg, zero, mean = signs("dU_eucap_eu")
    add("capping_cost_paired", mean, -0.195, 0.002)
    # Launch-set identity cannot be recovered from the aggregate JSON.
    capped = sensitive["seed23_full"]["eu_capped"]
    hybrid = sensitive["seed23_full"]["hybrid"]
    add("seed23_launch_count_equal", capped["launched_breaches"], hybrid["launched_breaches"], 0)
    add(
        "seed23_utility_not_identical",
        0 if abs(capped["utility"] - hybrid["utility"]) > 1e-12 else 1,
        0,
        0,
        note="Equal launch counts with unequal utility means the action sets differ off the launch margin.",
    )
    report = {"n_checks": len(checks), "n_failed": sum(not row["ok"] for row in checks), "checks": checks}
    (OUT / "published_output_validation.json").write_text(json.dumps(report, indent=2))
    print(f"failed {report['n_failed']} of {report['n_checks']}")
    for row in checks:
        if not row["ok"]:
            print("FAIL", row)


if __name__ == "__main__":
    main()
