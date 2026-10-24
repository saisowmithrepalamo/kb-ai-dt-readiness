"""Seed and payoff sensitivity analysis for KCID-AI.

Reuses the generator, rules and model fits in analysis.py. Besides the
policies in analysis.py it evaluates further baselines: knowledge-feature
boosting with the same rule layer, a direct expected-utility policy on the
knowledge logistic probabilities, that policy with the hard rules, the same
policy capped at KCID-AI's expected launched breaches, a rule-only policy, and
a hybrid that keeps KCID-AI's hard exclusions and rule explanations but sets
launches by the capped expected-utility policy. Each policy is also scored on
explanation coverage: the share of non-launched initiatives with a named rule.
Two analyses:

1. Seed replication: the population, split, fits and evaluation are
   regenerated for seeds 1..N_SEEDS (seed 23 is the reported run).
2. Payoff and policy sensitivity: on the seed-23 fits, one parameter at a
   time is varied (harden retention, harden cost, residual impact after
   hardening, soft-rule penalty, launch and defer thresholds).

Usage: python sensitivity.py [path/to/analysis.py directory]
Writes sensitivity_results.json, payoff_sensitivity.csv and Figure_3.png/.pdf
next to this file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE))
import analysis_published as A  # noqa: E402

N_SEEDS = 50
BASE = {"retain": 0.55, "h_cost": 5.0, "resid": 0.35, "penalty": 8.0, "t_launch": 10.0, "t_defer": -5.0}
GRID = {
    "retain": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    "h_cost": [0.0, 2.5, 5.0, 7.5, 10.0],
    "resid": [0.15, 0.25, 0.35, 0.45, 0.55],
    "penalty": [0.0, 4.0, 8.0, 12.0, 16.0],
    "t_launch": [5.0, 7.5, 10.0, 12.5, 15.0],
    "t_defer": [-15.0, -10.0, -5.0, 0.0],
}
LABEL = {
    "retain": "Harden value retention",
    "h_cost": "Harden cost",
    "resid": "Residual impact after hardening",
    "penalty": "Soft-rule penalty",
    "t_launch": "Launch threshold",
    "t_defer": "Defer threshold",
}
# name: (probability source, decision rule, rule layer)
# "threshold" uses Eq. (3) with the launch/defer thresholds; "eu" picks the
# action with the highest expected payoff under the evaluation payoffs;
# "eu_capped" is "eu" with a budget on expected launched breaches equal to
# KCID-AI's; "rules" ignores probabilities and acts on the rules alone.
# Rule layer: "off", "all" (hard exclusions and soft penalties) or "hard".
POLICIES = {
    "data_logistic": ("data_logistic", "threshold", "off"),
    "know_logistic": ("know_logistic", "threshold", "off"),
    "know_boost": ("know_boost", "threshold", "off"),
    "kcid": ("know_logistic", "threshold", "all"),
    "know_boost_rules": ("know_boost", "threshold", "all"),
    "eu_logistic": ("know_logistic", "eu", "off"),
    "eu_logistic_hard": ("know_logistic", "eu", "hard"),
    "eu_capped": ("know_logistic", "eu_capped", "off"),
    "rules_only": (None, "rules", "all"),
    "hybrid": ("know_logistic", "eu_capped", "hard"),
}
CONTRASTS = {
    "kcid_data": ("kcid", "data_logistic"),
    "know_data": ("know_logistic", "data_logistic"),
    "kcid_know": ("kcid", "know_logistic"),
    "kcid_boost": ("kcid", "know_boost"),
    "kcid_boostrules": ("kcid", "know_boost_rules"),
    "boostrules_boost": ("know_boost_rules", "know_boost"),
    "kcid_eu": ("kcid", "eu_logistic"),
    "euhard_eu": ("eu_logistic_hard", "eu_logistic"),
    "kcid_euhard": ("kcid", "eu_logistic_hard"),
    "kcid_eucap": ("kcid", "eu_capped"),
    "kcid_rulesonly": ("kcid", "rules_only"),
    "eucap_eu": ("eu_capped", "eu_logistic"),
    "kcid_hybrid": ("kcid", "hybrid"),
    "hybrid_eucap": ("hybrid", "eu_capped"),
}


def decide_eu(p, hard, value, impact, use_hard, prm, lam=0.0):
    """Direct expected-utility policy: argmax of expected launch, harden, defer payoffs.

    lam is a Lagrange price on each expected launched breach (0 = unconstrained).
    """
    e_launch = value - p * impact - lam * p
    e_harden = prm["retain"] * value - prm["h_cost"] - prm["resid"] * p * impact
    action = np.full(len(p), A.DEFER, dtype=int)
    best = np.zeros(len(p))
    action[e_harden > best] = A.HARDEN
    best = np.maximum(best, e_harden)
    action[e_launch > best] = A.LAUNCH
    if use_hard:
        action[hard] = A.DEFER
    return action


def decide_eu_capped(p, value, impact, prm, budget, iters=60, hard=None):
    """Expected-utility policy whose expected launched breaches, the sum of p over
    launches, do not exceed budget. The price lam is found by bisection; only
    predicted probabilities are used, never outcomes. If hard is given, those
    initiatives are deferred before the budget is applied (the hybrid policy)."""
    def expected_lb(lam):
        a = decide_eu(p, hard, value, impact, hard is not None, prm, lam)
        return float(p[a == A.LAUNCH].sum()), a
    e0, a0 = expected_lb(0.0)
    if e0 <= budget:
        return a0, 0.0
    lo, hi = 0.0, 1.0
    while expected_lb(hi)[0] > budget:
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if expected_lb(mid)[0] > budget:
            lo = mid
        else:
            hi = mid
    return expected_lb(hi)[1], hi


def decide_rules_only(hard, soft):
    """Rule-only policy: defer on a hard rule, harden on any soft rule, else launch."""
    action = np.full(len(hard), A.LAUNCH, dtype=int)
    action[soft > 0] = A.HARDEN
    action[hard] = A.DEFER
    return action


def decide(p, hard, soft, value, impact, use_rules, prm):
    penalty = soft * prm["penalty"] if use_rules else np.zeros(len(p))
    score = value - p * impact - penalty
    action = np.full(len(p), A.DEFER, dtype=int)
    action[score >= prm["t_defer"]] = A.HARDEN
    action[score >= prm["t_launch"]] = A.LAUNCH
    if use_rules:
        action[hard] = A.DEFER
    return action


def utility(action, breach, value, impact, prm):
    u = np.zeros(len(action))
    launch = action == A.LAUNCH
    harden = action == A.HARDEN
    u[launch] = value[launch] - breach[launch] * impact[launch]
    u[harden] = prm["retain"] * value[harden] - prm["h_cost"] - prm["resid"] * breach[harden] * impact[harden]
    return u


def build(seed: int):
    """Population, split and fitted probabilities, following analysis.main()."""
    df = A.generate(np.random.default_rng(seed))
    hard, soft = A.knowledge_rules(df)
    df["hard"], df["soft"] = hard.astype(int), soft
    idx = np.arange(A.N)
    rng_split = np.random.default_rng(seed)
    rng_split.shuffle(idx)
    b = idx[df.loc[idx, "breach"].to_numpy() == 1]
    s = idx[df.loc[idx, "breach"].to_numpy() == 0]
    nb, ns = int(A.TRAIN_FRAC * len(b)), int(A.TRAIN_FRAC * len(s))
    tr = np.concatenate([b[:nb], s[:ns]])
    te = np.concatenate([b[nb:], s[ns:]])
    rng_split.shuffle(tr)
    rng_split.shuffle(te)
    train = df.loc[tr].reset_index(drop=True)
    test = df.loc[te].reset_index(drop=True)
    cell = (test["data_vol"].to_numpy() >= train["data_vol"].median()) & (
        test["access"].to_numpy() <= np.quantile(train["access"], 0.40)
    )
    A.SEED = seed  # boosting random_state follows the replication seed
    probs = {
        "data_logistic": A.fit_proba(train, test, A.DATA_COLS, "logistic"),
        "know_logistic": A.fit_proba(train, test, A.ALL_COLS, "logistic"),
        "know_boost": A.fit_proba(train, test, A.ALL_COLS, "boost"),
    }
    probs["kcid"] = probs["know_logistic"]
    return test, cell, probs


def actions_for(test, probs, prm):
    v, imp = test["value"].to_numpy(), test["impact"].to_numpy()
    hard, soft = test["hard"].to_numpy().astype(bool), test["soft"].to_numpy()
    acts = {}
    for name, (src, kind, rules) in POLICIES.items():
        p = probs[src] if src else None
        if kind == "eu":
            acts[name] = decide_eu(p, hard, v, imp, rules == "hard", prm)
        elif kind == "rules":
            acts[name] = decide_rules_only(hard, soft)
        elif kind == "threshold":
            acts[name] = decide(p, hard, soft, v, imp, rules == "all", prm)
    # The capped policy's budget is KCID-AI's expected launched breaches.
    p_k = probs[POLICIES["kcid"][0]]
    budget = float(p_k[acts["kcid"] == A.LAUNCH].sum())
    acts["eu_capped"], _ = decide_eu_capped(probs["know_logistic"], v, imp, prm, budget)
    acts["hybrid"], _ = decide_eu_capped(probs["know_logistic"], v, imp, prm, budget, hard=hard)
    return {name: acts[name] for name in POLICIES}


def metrics(acts, test, cell, prm, probs=None):
    y = test["breach"].to_numpy()
    v, imp = test["value"].to_numpy(), test["impact"].to_numpy()
    hard, soft = test["hard"].to_numpy().astype(bool), test["soft"].to_numpy()
    out = {}
    for name, a in acts.items():
        u = utility(a, y, v, imp, prm)
        launch = a == A.LAUNCH
        row = {
            "utility": float(u.mean()),
            "launch_share": float(launch.mean()),
            "launched_breaches": int((launch & (y == 1)).sum()),
            "breach_among_launches": float(y[launch].mean()) if launch.any() else None,
            "cell_launched_breaches": int((launch & (y == 1) & cell).sum()),
            "n_breach": int(y.sum()),
            "cell_breaches": int((cell & (y == 1)).sum()),
        }
        # Explanation coverage: non-launched initiatives with at least one named rule,
        # and launches made while a soft rule flags a control gap.
        named = hard | (soft > 0)
        row["reason_coverage"] = float(named[~launch].mean()) if (~launch).any() else None
        row["flagged_launch_share"] = float((soft > 0)[launch].mean()) if launch.any() else None
        row["lb_rate"] = row["launched_breaches"] / row["n_breach"]
        row["cell_lb_rate"] = row["cell_launched_breaches"] / row["cell_breaches"]
        if probs is not None and POLICIES[name][0]:
            row["auc"] = A.auc_of(probs[POLICIES[name][0]], y)
        out[name] = row
    return out


def evaluate(test, cell, probs, prm):
    return metrics(actions_for(test, probs, prm), test, cell, prm)


def contrasts(m):
    out = {}
    for key, (left, right) in CONTRASTS.items():
        a, b = m[left], m[right]
        out["dU_" + key] = a["utility"] - b["utility"]
        out["dLB_" + key] = a["lb_rate"] - b["lb_rate"]
        out["dCell_" + key] = a["cell_lb_rate"] - b["cell_lb_rate"]
        if a.get("reason_coverage") is not None and b.get("reason_coverage") is not None:
            out["dRC_" + key] = a["reason_coverage"] - b["reason_coverage"]
            out["dFL_" + key] = a["flagged_launch_share"] - b["flagged_launch_share"]
    return out


def bootstrap_seed23(test, cell, probs, keys, n_boot=1000):
    """Paired bootstrap of the seed-23 test set, as in analysis.py (seed 24)."""
    acts = actions_for(test, probs, BASE)
    rng = np.random.default_rng(24)
    rows = []
    for _ in range(n_boot):
        d = rng.integers(0, len(test), len(test))
        sub = test.iloc[d].reset_index(drop=True)
        m = metrics({k: a[d] for k, a in acts.items()}, sub, cell[d], BASE)
        c = contrasts(m)
        rows.append({k: v for k, v in c.items() if k.split("_", 1)[1] in keys})
    df = pd.DataFrame(rows)
    return {
        col: {"ci95": [float(df[col].quantile(0.025)), float(df[col].quantile(0.975))]}
        for col in df.columns
    }


def summarise(rows):
    df = pd.DataFrame(rows)
    out = {}
    for col in df.columns:
        x = df[col].to_numpy()
        out[col] = {
            "mean": float(x.mean()),
            "p2_5": float(np.quantile(x, 0.025)),
            "p97_5": float(np.quantile(x, 0.975)),
            "share_positive": float((x > 0).mean()),
            "share_negative": float((x < 0).mean()),
        }
    return out


def main() -> None:
    seed_rows, seed_metrics = [], {}
    for seed in range(1, N_SEEDS + 1):
        test, cell, probs = build(seed)
        m = evaluate(test, cell, probs, BASE)
        seed_metrics[seed] = m
        seed_rows.append(contrasts(m))
    seed_summary = summarise(seed_rows)
    policy_rows = [
        {f"{name}.{k}": seed_metrics[s][name][k] for name in POLICIES for k in ("utility", "lb_rate", "cell_lb_rate", "reason_coverage", "flagged_launch_share")}
        for s in seed_metrics
    ]
    policy_summary = summarise(policy_rows)

    test, cell, probs = build(23)
    seed23_full = metrics(actions_for(test, probs, BASE), test, cell, BASE, probs)
    seed23_boot = bootstrap_seed23(test, cell, probs, {"kcid_boostrules", "kcid_eu", "boostrules_boost", "euhard_eu", "kcid_eucap", "kcid_rulesonly", "eucap_eu", "kcid_hybrid", "hybrid_eucap"})
    payoff = []
    for param, values in GRID.items():
        for val in values:
            prm = dict(BASE, **{param: val})
            c = contrasts(evaluate(test, cell, probs, prm))
            payoff.append({"parameter": param, "value": val, **c})

    results = {
        "n_seeds": N_SEEDS,
        "base": BASE,
        "seed23_check": seed_metrics[23],
        "seed23_full": seed23_full,
        "seed23_bootstrap": seed23_boot,
        "seed_summary": seed_summary,
        "policy_summary": policy_summary,
        "seed_rows": seed_rows,
        "payoff": payoff,
    }
    (HERE / "sensitivity_results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(payoff).to_csv(HERE / "payoff_sensitivity.csv", index=False)
    _figure(payoff)
    print(json.dumps({"seed23": seed_metrics[23], "seed_summary": seed_summary}, indent=2))


def _figure(payoff) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "axes.linewidth": 0.8})
    df = pd.DataFrame(payoff)
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.8), sharey=True)
    for ax, param in zip(axes.ravel(), GRID):
        d = df[df["parameter"] == param]
        ax.axhline(0, color="#888888", lw=0.8)
        ax.axvline(BASE[param], color="#BBBBBB", lw=0.8, ls=":")
        ax.plot(d["value"], d["dU_know_data"], "o-", color="#0072B2", ms=4, lw=1.5, label="Features: knowledge − data logistic")
        ax.plot(d["value"], d["dU_kcid_know"], "D-", color="#009E73", ms=4, lw=1.5, label="Rules: KCID-AI − knowledge logistic")
        ax.set_title(LABEL[param], fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8, length=3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean utility difference")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(HERE / "Figure_3.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(HERE / "Figure_3.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
