"""Decision-theoretic checks and learned-rule baselines for KCID-AI (second revision).

Reuses analysis.py, sensitivity.py and extensions.py unchanged. Three analyses:

1. Oracle bound and decomposition. The expected-utility (EU) policy evaluated with the
   generator's true breach probability q is the best any policy can do. The expected-
   utility gap between it and KCID-AI splits into probability error, heuristic
   thresholds and the rule layer.
2. Numerical checks of the propositions on every test initiative: the oracle EU policy
   is pointwise optimal; a plug-in EU policy loses at most I*|p - q|; and a cautious
   override of an EU action gains in expectation only where the model underestimates
   risk (p < q), by at most I*(q - p).
3. Learned rule layers with the expert layer's mechanism and coverage: rules read off a
   shallow decision tree fitted to training breaches ("learned") or to the training
   residual of out-of-fold knowledge-logistic probabilities ("residual"), compared with
   the expert rules r1-r7 under KCID-AI's thresholds and under the EU policy. A graded
   variant ("rescorr") adds each positive residual leaf's mean residual to the risk
   estimate instead of forcing deferral or a fixed penalty.

Usage: python revision.py   (writes revision_results.json and revision_seeds.csv)
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import analysis as A
import extensions as E
import sensitivity as S

HERE = Path(__file__).resolve().parent
SEEDS = range(1, S.N_SEEDS + 1)
# Chosen because it brings learned-rule coverage closest to the expert layer's on seeds
# 1-10 (depth 3 with 100 per leaf and depth 4 with 75 were the alternatives), not on outcomes.
TREE = {"max_depth": 4, "min_samples_leaf": 50}
TOL = 1e-9
RULE_SETS = ("expert", "learned", "residual", "rescorr")
NAMES = {"inn": "innovation", "ent": "entrepreneurship", "data_vol": "data volume", "data_sens": "data sensitivity",
         "dt": "DT breadth", "access": "access control", "crypto": "encryption", "monitor": "monitoring",
         "respond": "incident response", "skills": "security skills", "vendor": "vendor control"}
CONTRASTS = {
    "expert_none": ("kcid_expert", "know_logistic"),
    "learned_none": ("kcid_learned", "know_logistic"),
    "residual_none": ("kcid_residual", "know_logistic"),
    "learned_expert": ("kcid_learned", "kcid_expert"),
    "residual_expert": ("kcid_residual", "kcid_expert"),
    "residual_learned": ("kcid_residual", "kcid_learned"),
    "rescorr_none": ("kcid_rescorr", "know_logistic"),
    "rescorr_expert": ("kcid_rescorr", "kcid_expert"),
    "euexpert_eu": ("eu_expert", "eu_logistic"),
    "eulearned_eu": ("eu_learned", "eu_logistic"),
    "euresidual_eu": ("eu_residual", "eu_logistic"),
    "eurescorr_eu": ("eu_rescorr", "eu_logistic"),
    "eurescorr_euexpert": ("eu_rescorr", "eu_expert"),
    "oracle_kcid": ("oracle_eu", "kcid_expert"),
}


def expected_payoff(action, q, value, impact, prm=S.BASE):
    """Expected evaluation payoff of each action at breach probability q."""
    e = np.zeros(len(action))
    launch, harden = action == A.LAUNCH, action == A.HARDEN
    e[launch] = value[launch] - q[launch] * impact[launch]
    e[harden] = prm["retain"] * value[harden] - prm["h_cost"] - prm["resid"] * q[harden] * impact[harden]
    return e


def corrected(p, soft, impact, prm=S.BASE):
    """A soft penalty P in J = V - pI - P is the same as raising p by P/I."""
    return np.clip(p + prm["penalty"] * soft / impact, 0.0, 1.0)


def leaf_conditions(tree, cols) -> dict[int, str]:
    """The conjunction of split conditions that leads to each leaf."""
    t, out = tree.tree_, {}

    def walk(node, bounds):
        if t.children_left[node] == -1:
            parts = []
            for f, (lo, hi) in bounds.items():
                name = NAMES[cols[f]]
                if lo is not None and hi is not None:
                    parts.append(f"{lo:.2f} < {name} <= {hi:.2f}")
                elif lo is not None:
                    parts.append(f"{name} > {lo:.2f}")
                else:
                    parts.append(f"{name} <= {hi:.2f}")
            out[node] = " and ".join(parts)
            return
        f, thr = t.feature[node], float(t.threshold[node])
        lo, hi = bounds.get(f, (None, None))
        walk(t.children_left[node], {**bounds, f: (lo, thr if hi is None else min(hi, thr))})
        walk(t.children_right[node], {**bounds, f: (thr if lo is None else max(lo, thr), hi)})

    walk(0, {})
    return out


def select_leaves(score, cov, eligible, hard_target, soft_target):
    """Highest-scoring leaves become hard rules, the next ones soft rules; each count is
    chosen so that training coverage comes closest to the expert layer's."""
    order = sorted(eligible, key=lambda leaf: -score[leaf])

    def take(cands, target):
        best_k, best_err, cum = 0, target, 0.0
        for k, leaf in enumerate(cands, 1):
            cum += cov[leaf]
            if abs(cum - target) < best_err:
                best_k, best_err = k, abs(cum - target)
        return cands[:best_k]

    hard = take(order, hard_target)
    return hard, take(order[len(hard):], soft_target)


def learn_rules(train, test, seed, target):
    """A rule layer read off a shallow tree, with the expert layer's training coverage.

    Also returns, for the residual tree, the graded correction of each test initiative:
    the mean training residual of its leaf where that is positive, else 0.
    """
    x, y = train[A.ALL_COLS], train["breach"].to_numpy()
    hard_target = float(train["hard"].mean())
    soft_target = float(((train["soft"] > 0) & (train["hard"] == 0)).mean())
    if target == "learned":
        tree = DecisionTreeClassifier(**TREE, random_state=seed).fit(x, y)
        signal = y.astype(float)
    else:
        # Out-of-fold knowledge-logistic probabilities, so the residual is not fitted in-sample.
        logit = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"))
        p_oof = cross_val_predict(logit, x, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                  method="predict_proba")[:, 1]
        signal = y - p_oof
        tree = DecisionTreeRegressor(**TREE, random_state=seed).fit(x, signal)
    leaf_tr = tree.apply(x)
    leaves = np.unique(leaf_tr)
    score = {int(lf): float(signal[leaf_tr == lf].mean()) for lf in leaves}
    cov = {int(lf): float((leaf_tr == lf).mean()) for lf in leaves}
    # Caution can only pay where risk is underestimated, so residual rules use positive leaves only.
    eligible = list(score) if target == "learned" else [lf for lf in score if score[lf] > 0]
    hard_l, soft_l = select_leaves(score, cov, eligible, hard_target, soft_target)
    leaf_te = tree.apply(test[A.ALL_COLS])
    text = leaf_conditions(tree, A.ALL_COLS)
    positive = [lf for lf in score if score[lf] > 0]
    info = {
        "hard_target": hard_target, "soft_target": soft_target,
        "hard_coverage_train": sum(cov[lf] for lf in hard_l), "soft_coverage_train": sum(cov[lf] for lf in soft_l),
        "n_hard": len(hard_l), "n_soft": len(soft_l), "n_leaves": len(score),
        "rules": [{"type": "hard" if lf in hard_l else "soft", "condition": text[lf],
                   "train_coverage": cov[lf], "train_signal": score[lf]} for lf in hard_l + soft_l],
    }
    delta = None
    if target == "residual":
        delta = np.array([max(score[int(lf)], 0.0) for lf in leaf_te])
        info["positive_coverage_train"] = sum(cov[lf] for lf in positive)
        info["corrections"] = [{"condition": text[lf], "train_coverage": cov[lf], "correction": score[lf]}
                               for lf in sorted(positive, key=lambda lf: -score[lf])]
    return np.isin(leaf_te, hard_l), np.isin(leaf_te, soft_l).astype(int), info, delta


def policy_actions(test, probs, rule_sets, risks, prm=S.BASE):
    """risks[name] is the risk estimate the EU policy uses with that rule set."""
    v, imp = test["value"].to_numpy(), test["impact"].to_numpy()
    p, q = probs["know_logistic"], test["p_true"].to_numpy()
    none_h, none_s = np.zeros(len(p), dtype=bool), np.zeros(len(p), dtype=int)
    acts = {
        "know_logistic": S.decide(p, none_h, none_s, v, imp, False, prm),
        "eu_logistic": S.decide_eu(p, none_h, v, imp, False, prm),
        "oracle_eu": S.decide_eu(q, none_h, v, imp, False, prm),
    }
    for name, (hard, soft) in rule_sets.items():
        if name == "rescorr":
            # Graded corrections enter the score as risk, J = V - (p + delta)I, with no hard rules.
            acts["kcid_rescorr"] = S.decide(risks[name], none_h, none_s, v, imp, False, prm)
        else:
            acts[f"kcid_{name}"] = S.decide(p, hard, soft, v, imp, True, prm)
        acts[f"eu_{name}"] = S.decide_eu(risks[name], hard, v, imp, True, prm)
    return acts


def _rule_set_of(policy):
    tail = policy.split("_", 1)[1]
    return tail if tail in RULE_SETS else "expert"  # policies without rules are scored on the expert flags


def policy_metrics(acts, test, cell, rule_sets, prm=S.BASE):
    y, q = test["breach"].to_numpy(), test["p_true"].to_numpy()
    v, imp = test["value"].to_numpy(), test["impact"].to_numpy()
    out = {}
    for name, a in acts.items():
        hard, soft = rule_sets[_rule_set_of(name)]
        launch = a == A.LAUNCH
        named = hard | (soft > 0)
        out[name] = {
            "utility": float(S.utility(a, y, v, imp, prm).mean()),
            "expected_utility": float(expected_payoff(a, q, v, imp, prm).mean()),
            "launch_share": float(launch.mean()),
            "launched_breaches": int((launch & (y == 1)).sum()),
            "lb_rate": float((launch & (y == 1)).sum() / y.sum()),
            "cell_lb_rate": float((launch & (y == 1) & cell).sum() / (cell & (y == 1)).sum()),
            "reason_coverage": float(named[~launch].mean()) if (~launch).any() else None,
            "flagged_launch_share": float((soft > 0)[launch].mean()) if launch.any() else None,
        }
    return out


def contrasts(m):
    out = {}
    for key, (left, right) in CONTRASTS.items():
        out[f"dU_{key}"] = m[left]["utility"] - m[right]["utility"]
        out[f"dEU_{key}"] = m[left]["expected_utility"] - m[right]["expected_utility"]
        out[f"dLB_{key}"] = m[left]["lb_rate"] - m[right]["lb_rate"]
    return out


def check_propositions(acts, test, probs, rule_sets, risks, prm=S.BASE):
    """Largest violation of each proposition (all must be <= TOL), plus override diagnostics."""
    v, imp, q = test["value"].to_numpy(), test["impact"].to_numpy(), test["p_true"].to_numpy()
    p = probs["know_logistic"]
    none_h = np.zeros(len(p), dtype=bool)
    e = {k: expected_payoff(a, q, v, imp, prm) for k, a in acts.items()}
    out = {"prop1": max(float((e[k] - e["oracle_eu"]).max()) for k in e)}
    # Proposition 2 for the plug-in EU policy at p and at each rule-corrected risk estimate.
    out["prop2"] = max(
        float((e["oracle_eu"] - expected_payoff(S.decide_eu(r, none_h, v, imp, False, prm), q, v, imp, prm)
               - imp * np.abs(r - q)).max()) for r in [p, *risks.values()])
    out["prop3"] = 0.0
    for name in rule_sets:
        for base, ruled in (("eu_logistic", f"eu_{name}"), ("know_logistic", f"kcid_{name}")):
            changed = acts[ruled] != acts[base]
            gain = e[ruled] - e[base]
            under = p < q
            row = {
                "n_changed": int(changed.sum()),
                "share_changed_underestimated": float(under[changed].mean()) if changed.any() else None,
                "expected_gain_per_initiative": float(gain.mean()),
                "gain_where_underestimated": float(gain[changed & under].sum() / len(p)),
                "gain_where_not_underestimated": float(gain[changed & ~under].sum() / len(p)),
            }
            if base == "eu_logistic":
                # The EU overrides are cautious by construction (higher risk or forced deferral).
                assert (acts[ruled] <= acts[base]).all(), "an EU rule override was not cautious"
                bound = imp * np.clip(q - p, 0.0, None)
                row["bound_per_initiative"] = float(bound[changed].sum() / len(p))
                out["prop3"] = max(out["prop3"], float((gain - bound)[changed].max()) if changed.any() else 0.0)
            out[f"{ruled}_vs_{base}"] = row
    return out


def run_seed(seed: int) -> dict:
    train, test, cell, probs = E.build_full(seed)
    p, imp = probs["know_logistic"], test["impact"].to_numpy()
    rule_sets = {"expert": (test["hard"].to_numpy().astype(bool), test["soft"].to_numpy())}
    info = {}
    for target in ("learned", "residual"):
        hard, soft, info[target], delta = learn_rules(train, test, seed, target)
        rule_sets[target] = (hard, soft)
    # The graded corrections flag their leaves as named reasons but never force deferral.
    rule_sets["rescorr"] = (np.zeros(len(p), dtype=bool), (delta > 0).astype(int))
    risks = {n: corrected(p, s, imp) for n, (_, s) in rule_sets.items() if n != "rescorr"}
    risks["rescorr"] = np.clip(p + delta, 0.0, 1.0)
    acts = policy_actions(test, probs, rule_sets, risks)
    m = policy_metrics(acts, test, cell, rule_sets)
    checks = check_propositions(acts, test, probs, rule_sets, risks)
    row = {"seed": seed, **contrasts(m),
           **{f"{k}_{pol}": v for pol, d in m.items() for k, v in d.items()},
           **{f"prop{i}_violation": checks[f"prop{i}"] for i in (1, 2, 3)},
           **{f"{t}_{k}": info[t][k] for t in info for k in ("hard_coverage_train", "soft_coverage_train", "n_hard", "n_soft")},
           "rescorr_positive_coverage_train": info["residual"]["positive_coverage_train"],
           "rescorr_mean_correction_test": float(delta.mean()),
           **{f"diag_{d}_{k}": v for d, r in checks.items() if isinstance(r, dict) for k, v in r.items()}}
    if seed == 23:
        row["metrics"], row["rules"], row["checks"] = m, info, checks
        rng = np.random.default_rng(24)  # paired bootstrap as in analysis.py
        boot = []
        for _ in range(A.N_BOOT):
            d = rng.integers(0, len(test), len(test))
            sub = {k: (h[d], s[d]) for k, (h, s) in rule_sets.items()}
            boot.append(contrasts(policy_metrics({k: a[d] for k, a in acts.items()},
                                                 test.iloc[d].reset_index(drop=True), cell[d], sub)))
        bdf = pd.DataFrame(boot)
        row["bootstrap"] = {c: [float(bdf[c].quantile(0.025)), float(bdf[c].quantile(0.975))] for c in bdf.columns}
    return row


def summarise(rows: list[dict]) -> dict:
    d = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows])
    out = {}
    for col in d.columns:
        if col == "seed" or d[col].isna().all():
            continue
        x = d[col].astype(float)
        out[col] = {"mean": float(x.mean()), "p2_5": float(x.quantile(0.025)), "p97_5": float(x.quantile(0.975)),
                    "min": float(x.min()), "max": float(x.max()), "n_pos": int((x > 0).sum()), "n_neg": int((x < 0).sum())}
    return out


def main() -> None:
    # The pipeline must reproduce the published seed-23 run exactly.
    _, test, cell, probs = E.build_full(23)
    published = json.loads((HERE / "sensitivity_results.json").read_text())["seed23_check"]
    check = S.evaluate(test, cell, probs, S.BASE)
    assert all(abs(check[k]["utility"] - published[k]["utility"]) < 1e-12 for k in published), "seed-23 run drifted"

    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(run_seed, SEEDS))
    s23 = next(r for r in rows if r["seed"] == 23)
    assert abs(s23["metrics"]["kcid_expert"]["utility"] - published["kcid"]["utility"]) < 1e-12, "KCID-AI drifted"
    assert abs(s23["metrics"]["eu_logistic"]["utility"] - published["eu_logistic"]["utility"]) < 1e-12, "EU drifted"
    worst = {i: max(r[f"prop{i}_violation"] for r in rows) for i in (1, 2, 3)}
    assert all(w <= TOL for w in worst.values()), f"a proposition check failed: {worst}"

    results = {
        "tree": TREE, "n_seeds": len(rows), "tolerance": TOL, "worst_violation": worst,
        "seed23": {"metrics": s23["metrics"], "rules": s23["rules"], "checks": s23["checks"],
                   "contrasts": {k: v for k, v in s23.items() if k.startswith(("dU_", "dEU_", "dLB_"))},
                   "bootstrap": s23["bootstrap"]},
        "seeds": summarise(rows),
    }
    (HERE / "revision_results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]).to_csv(
        HERE / "revision_seeds.csv", index=False)
    print(json.dumps({"worst_violation": worst, "seed23_contrasts": results["seed23"]["contrasts"]}, indent=1))


if __name__ == "__main__":
    main()
