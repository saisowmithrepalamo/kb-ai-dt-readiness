"""Additional robustness analyses for KCID-AI (revision).

Reuses analysis.py and sensitivity.py unchanged. Four analyses:

1. Calibration of the probability models: Brier score, calibration-in-the-large,
   calibration slope, expected calibration error and error against the true
   probability, on the seed-23 test set and across seeds 1..50.
2. A cross-validated gradient-boosting baseline: 5-fold grid search on the
   training sample (log loss), with and without the rule layer, for seeds 1..50.
3. Alternative generator structures: the breach process is changed while the
   indicators, split, rules, models and policies stay as in the paper.
4. Joint sensitivity: 500 settings of the six payoff and policy constants
   drawn jointly on the seed-23 fits.

Usage: python extensions.py   (writes extensions_results.json, Figure_4.png/.pdf)
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import analysis as A
import sensitivity as S

HERE = Path(__file__).resolve().parent
SEEDS = range(1, S.N_SEEDS + 1)
N_JOINT = 500
JOINT_SEED = 2026
GB_GRID = {"n_estimators": [100, 250, 500], "learning_rate": [0.02, 0.05, 0.1], "max_depth": [2, 3]}
MODELS = ("data_logistic", "know_logistic", "know_boost")

# Breach-process coefficients. "base" is the published generator (analysis.py).
BASE_COEF = {
    "sens": 0.35, "vol": 0.25, "dt": 0.15, "ent": 0.10,
    "access": -0.10, "monitor": -0.08, "crypto": 0.0, "respond": 0.0, "skills": 0.0, "vendor": 0.0,
    "gap_access": 3.10, "gap_mon": 2.15, "gap_crypto": 1.80, "venture": 1.35,
}
VARIANTS = {
    "base": {},
    "gaps_half": {"gap_access": 1.55, "gap_mon": 1.075, "gap_crypto": 0.90},
    "gaps_x1.5": {"gap_access": 4.65, "gap_mon": 3.225, "gap_crypto": 2.70},
    "no_venture_gap": {"venture": 0.0},
    "linear_controls": {"gap_access": 0.0, "gap_mon": 0.0, "gap_crypto": 0.0,
                        "access": -1.60, "monitor": -1.20, "crypto": -0.80,
                        "respond": -0.40, "skills": -0.40, "vendor": -0.40},
    "no_control_signal": {"gap_access": 0.0, "gap_mon": 0.0, "gap_crypto": 0.0, "venture": 0.0,
                          "access": 0.0, "monitor": 0.0},
}


def generate_variant(rng: np.random.Generator, coef: dict) -> pd.DataFrame:
    """analysis.generate with the breach coefficients exposed. Random draws are
    taken in the same order, so every variant shares the published indicators."""
    n = A.N
    sig = A._sigmoid
    z_innov, z_ent, z_data, z_ctrl = (rng.normal(size=n) for _ in range(4))
    z_dt = 0.40 * z_innov + 0.30 * z_data + rng.normal(scale=0.85, size=n)
    inn = sig(0.85 * z_innov + 0.20 * rng.normal(size=n))
    ent = sig(0.80 * z_ent + 0.15 * z_innov + 0.20 * rng.normal(size=n))
    data_vol = sig(0.90 * z_data + 0.15 * rng.normal(size=n))
    data_sens = sig(0.45 * z_data + 0.85 * rng.normal(size=n))
    dt = sig(z_dt)
    access = sig(0.95 * z_ctrl + 0.10 * z_innov + 0.25 * rng.normal(size=n))
    crypto = sig(0.80 * z_ctrl + 0.25 * rng.normal(size=n))
    monitor = sig(0.85 * z_ctrl + 0.05 * z_data + 0.30 * rng.normal(size=n))
    respond = sig(0.75 * z_ctrl + 0.28 * rng.normal(size=n))
    skills = sig(0.50 * z_ctrl + 0.20 * z_innov + 0.40 * rng.normal(size=n))
    vendor = sig(0.65 * z_ctrl + 0.30 * rng.normal(size=n))
    gap_access = np.clip(data_sens - access, 0, None)
    gap_mon = np.clip(data_vol - monitor, 0, None)
    gap_crypto = np.clip(data_sens - crypto, 0, None) * (data_sens > 0.75)
    venture_gap = ((ent > 0.70) & (skills < 0.45) & (data_sens > 0.50)).astype(float)
    noise = rng.normal(size=n)
    c = coef
    slope = (c["sens"] * data_sens + c["vol"] * data_vol + c["dt"] * dt + c["ent"] * ent
             + c["access"] * access + c["monitor"] * monitor + c["crypto"] * crypto
             + c["respond"] * respond + c["skills"] * skills + c["vendor"] * vendor
             + c["gap_access"] * gap_access + c["gap_mon"] * gap_mon + c["gap_crypto"] * gap_crypto
             + c["venture"] * venture_gap + 0.25 * noise)
    lo, hi = -8.0, 2.0
    for _ in range(40):  # same intercept calibration as analysis.py: mean p = 0.36
        mid = 0.5 * (lo + hi)
        if float(sig(mid + slope).mean()) > 0.36:
            hi = mid
        else:
            lo = mid
    p_true = sig(0.5 * (lo + hi) + slope)
    breach = rng.random(n) < p_true
    return pd.DataFrame({
        "inn": inn, "ent": ent, "data_vol": data_vol, "data_sens": data_sens, "dt": dt,
        "access": access, "crypto": crypto, "monitor": monitor, "respond": respond,
        "skills": skills, "vendor": vendor, "p_true": p_true, "breach": breach.astype(int),
        "impact": 25.0 + 90.0 * data_sens + 40.0 * data_vol + 20.0 * dt,
        "value": 18.0 + 35.0 * inn + 22.0 * ent + 12.0 * dt,
        "venture_gap": venture_gap.astype(int),
    })


def build_full(seed: int, coef: dict | None = None):
    """sensitivity.build, also returning the training sample."""
    rng = np.random.default_rng(seed)
    df = A.generate(rng) if coef is None else generate_variant(rng, coef)
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
    A.SEED = seed
    probs = {
        "data_logistic": A.fit_proba(train, test, A.DATA_COLS, "logistic"),
        "know_logistic": A.fit_proba(train, test, A.ALL_COLS, "logistic"),
        "know_boost": A.fit_proba(train, test, A.ALL_COLS, "boost"),
    }
    probs["kcid"] = probs["know_logistic"]
    return train, test, cell, probs


def calibration(p: np.ndarray, y: np.ndarray, p_true: np.ndarray, n_bins: int = 10) -> dict:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(q / (1 - q)).reshape(-1, 1)
    fit = LogisticRegression(C=1e10, max_iter=1000).fit(lp, y)  # effectively unpenalised
    bins = np.array_split(np.argsort(p), n_bins)
    ece = sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in bins)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "citl": float(p.mean() - y.mean()),
        "slope": float(fit.coef_[0, 0]),
        "ece": float(ece),
        "mae_true": float(np.mean(np.abs(p - p_true))),
        "bins": [{"p_mean": float(p[b].mean()), "y_rate": float(y[b].mean()),
                  "p_true": float(p_true[b].mean()), "n": int(len(b))} for b in bins],
    }


def _contrast(m: dict, left: str, right: str) -> dict:
    a, b = m[left], m[right]
    return {"dU": a["utility"] - b["utility"], "dLB": a["lb_rate"] - b["lb_rate"],
            "dCell": a["cell_lb_rate"] - b["cell_lb_rate"]}


TUNED_CONTRASTS = {
    "kcid_tuned": ("kcid", "tuned_boost"),
    "kcid_tunedrules": ("kcid", "tuned_boost_rules"),
    "tuned_fixed": ("tuned_boost", "know_boost"),
    "tunedrules_tuned": ("tuned_boost_rules", "tuned_boost"),
}


def tuned_actions(test, probs, p_tuned):
    acts = S.actions_for(test, probs, S.BASE)
    v, imp = test["value"].to_numpy(), test["impact"].to_numpy()
    hard, soft = test["hard"].to_numpy().astype(bool), test["soft"].to_numpy()
    acts["tuned_boost"] = S.decide(p_tuned, hard, soft, v, imp, False, S.BASE)
    acts["tuned_boost_rules"] = S.decide(p_tuned, hard, soft, v, imp, True, S.BASE)
    return acts


def run_tuned(seed: int) -> dict:
    """Cross-validated boosting for one replication seed."""
    train, test, cell, probs = build_full(seed)
    search = GridSearchCV(
        GradientBoostingClassifier(subsample=0.85, random_state=seed), GB_GRID,
        scoring="neg_log_loss", cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=1,
    ).fit(train[A.ALL_COLS], train["breach"])
    p_tuned = search.best_estimator_.predict_proba(test[A.ALL_COLS])[:, 1]
    acts = tuned_actions(test, probs, p_tuned)
    m = S.metrics(acts, test, cell, S.BASE)
    y = test["breach"].to_numpy()
    row = {"seed": seed, "best": search.best_params_,
           "auc_tuned": A.auc_of(p_tuned, y), "auc_fixed": A.auc_of(probs["know_boost"], y),
           "brier_tuned": float(np.mean((p_tuned - y) ** 2)),
           "brier_fixed": float(np.mean((probs["know_boost"] - y) ** 2)),
           "tuned_boost": m["tuned_boost"], "tuned_boost_rules": m["tuned_boost_rules"]}
    for key, (left, right) in TUNED_CONTRASTS.items():
        for k, v in _contrast(m, left, right).items():
            row[f"{k}_{key}"] = v
    if seed == 23:
        rng = np.random.default_rng(24)  # paired bootstrap as in analysis.py
        boot = []
        for _ in range(A.N_BOOT):
            d = rng.integers(0, len(test), len(test))
            mb = S.metrics({k: a[d] for k, a in acts.items()}, test.iloc[d].reset_index(drop=True), cell[d], S.BASE)
            boot.append({f"{k}_{key}": v for key, (l, r) in TUNED_CONTRASTS.items() for k, v in _contrast(mb, l, r).items()})
        bdf = pd.DataFrame(boot)
        row["bootstrap"] = {c: [float(bdf[c].quantile(0.025)), float(bdf[c].quantile(0.975))] for c in bdf.columns}
    return row


def run_variant(args) -> dict:
    """One generator variant and one replication seed."""
    name, seed = args
    coef = dict(BASE_COEF, **VARIANTS[name])
    _, test, cell, probs = build_full(seed, coef)
    m = S.evaluate(test, cell, probs, S.BASE)
    y, pt = test["breach"].to_numpy(), test["p_true"].to_numpy()
    row = {"variant": name, "seed": seed, **S.contrasts(m),
           "auc_know": A.auc_of(probs["know_logistic"], y), "auc_data": A.auc_of(probs["data_logistic"], y)}
    for model in MODELS:
        cal = calibration(probs[model], y, pt)
        row.update({f"{k}_{model}": v for k, v in cal.items() if k != "bins"})
        if name == "base" and seed == 23:
            row[f"bins_{model}"] = cal["bins"]
    return row


def run_joint() -> list[dict]:
    """Joint draws of the six constants, uniform on the one-at-a-time ranges."""
    _, test, cell, probs = build_full(23)
    rng = np.random.default_rng(JOINT_SEED)
    rows = []
    for _ in range(N_JOINT):
        prm = {k: float(rng.uniform(min(v), max(v))) for k, v in S.GRID.items()}
        rows.append({**prm, **S.contrasts(S.evaluate(test, cell, probs, prm))})
    return rows


def _share(x, pred) -> int:
    return int(pred(np.asarray(x)).sum())


def summarise_variants(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    out = {}
    for name, d in df.groupby("variant", sort=False):
        out[name] = {
            "n": int(len(d)),
            "feature_dU_mean": float(d["dU_know_data"].mean()), "feature_dU_pos": _share(d["dU_know_data"], lambda x: x > 0),
            "rule_dU_mean": float(d["dU_kcid_know"].mean()), "rule_dU_pos": _share(d["dU_kcid_know"], lambda x: x > 0),
            "rule_dLB_mean": float(d["dLB_kcid_know"].mean()), "rule_dLB_neg": _share(d["dLB_kcid_know"], lambda x: x < 0),
            "kcid_eucap_dU_mean": float(d["dU_kcid_eucap"].mean()), "eucap_better": _share(d["dU_kcid_eucap"], lambda x: x < 0),
            "kcid_eu_dU_mean": float(d["dU_kcid_eu"].mean()),
            "kcid_rulesonly_dU_pos": _share(d["dU_kcid_rulesonly"], lambda x: x > 0),
            "auc_know_mean": float(d["auc_know"].mean()), "auc_data_mean": float(d["auc_data"].mean()),
            "brier_know_mean": float(d["brier_know_logistic"].mean()),
            "ece_know_mean": float(d["ece_know_logistic"].mean()),
            "slope_know_mean": float(d["slope_know_logistic"].mean()),
        }
    return out


def summarise_calibration(rows: list[dict]) -> dict:
    d = pd.DataFrame([r for r in rows if r["variant"] == "base"])
    out = {}
    for model in MODELS:
        out[model] = {k: {"mean": float(d[f"{k}_{model}"].mean()),
                          "p2_5": float(d[f"{k}_{model}"].quantile(0.025)),
                          "p97_5": float(d[f"{k}_{model}"].quantile(0.975))}
                      for k in ("brier", "citl", "slope", "ece", "mae_true")}
        s23 = next(r for r in rows if r["variant"] == "base" and r["seed"] == 23)
        out[model]["seed23"] = {k: s23[f"{k}_{model}"] for k in ("brier", "citl", "slope", "ece", "mae_true")}
    return out


def summarise_tuned(rows: list[dict]) -> dict:
    d = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows])
    out = {"best_params": pd.Series([json.dumps(r["best"], sort_keys=True) for r in rows]).value_counts().to_dict()}
    for col in [c for c in d.columns if c.startswith(("dU_", "dLB_", "dCell_", "auc_", "brier_"))]:
        out[col] = {"mean": float(d[col].mean()), "p2_5": float(d[col].quantile(0.025)),
                    "p97_5": float(d[col].quantile(0.975)), "n_pos": int((d[col] > 0).sum()),
                    "n_neg": int((d[col] < 0).sum())}
    s23 = next(r for r in rows if r["seed"] == 23)
    out["seed23"] = {k: v for k, v in s23.items() if k != "bootstrap"}
    out["seed23_bootstrap"] = s23["bootstrap"]
    return out


def summarise_joint(rows: list[dict]) -> dict:
    d = pd.DataFrame(rows)
    out = {"n": int(len(d))}
    for col, pred, label in [
        ("dU_know_data", lambda x: x > 0, "feature_dU_pos"), ("dU_kcid_know", lambda x: x > 0, "rule_dU_pos"),
        ("dLB_kcid_know", lambda x: x < 0, "rule_dLB_neg"), ("dLB_kcid_know", lambda x: x > 0, "rule_dLB_pos"),
        ("dU_kcid_eu", lambda x: x < 0, "eu_better"), ("dU_kcid_eucap", lambda x: x < 0, "eucap_better"),
        ("dU_kcid_rulesonly", lambda x: x > 0, "kcid_beats_rulesonly"),
        ("dRC_kcid_hybrid", lambda x: x > 0, "kcid_higher_coverage"), ("dFL_kcid_hybrid", lambda x: x < 0, "kcid_fewer_flags"),
    ]:
        out[label] = _share(d[col], pred)
    for col in ("dU_know_data", "dU_kcid_know", "dLB_kcid_know", "dU_kcid_eucap"):
        out[col] = {"mean": float(d[col].mean()), "min": float(d[col].min()), "max": float(d[col].max()),
                    "p2_5": float(d[col].quantile(0.025)), "p97_5": float(d[col].quantile(0.975))}
    out["spearman_rule_dU"] = {p: float(d[p].corr(d["dU_kcid_know"], method="spearman")) for p in S.GRID}
    return out


def _figure(bins: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "axes.linewidth": 0.8})
    style = {"data_logistic": ("s--", "#D55E00", "Data-only logistic"),
             "know_logistic": ("o-", "#0072B2", "Knowledge logistic (KCID-AI score)"),
             "know_boost": ("^-", "#7F7F7F", "Knowledge boosting")}
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8, ls=":", label="Perfect calibration")
    for model, (fmt, color, label) in style.items():
        b = bins[model]
        ax.plot([r["p_mean"] for r in b], [r["y_rate"] for r in b], fmt, color=color, ms=4.5, lw=1.4, label=label)
    ax.set_xlabel("Mean predicted breach probability (decile of predictions)")
    ax.set_ylabel("Observed breach rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "Figure_4.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(HERE / "Figure_4.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    # The base variant must reproduce the published seed-23 run exactly.
    _, test, cell, probs = build_full(23, dict(BASE_COEF))
    published = json.loads((HERE / "sensitivity_results.json").read_text())["seed23_check"]
    check = S.evaluate(test, cell, probs, S.BASE)
    assert all(abs(check[k]["utility"] - published[k]["utility"]) < 1e-12 for k in published), "base variant drifted"

    with ProcessPoolExecutor() as pool:
        variant_rows = list(pool.map(run_variant, [(v, s) for v in VARIANTS for s in SEEDS]))
        tuned_rows = list(pool.map(run_tuned, SEEDS))
    joint_rows = run_joint()

    base23 = next(r for r in variant_rows if r["variant"] == "base" and r["seed"] == 23)
    _figure({m: base23[f"bins_{m}"] for m in MODELS})
    results = {
        "variants": {k: dict(BASE_COEF, **v) for k, v in VARIANTS.items()},
        "gb_grid": GB_GRID, "n_joint": N_JOINT, "joint_seed": JOINT_SEED,
        "calibration": summarise_calibration(variant_rows),
        "calibration_bins_seed23": {m: base23[f"bins_{m}"] for m in MODELS},
        "tuned_boosting": summarise_tuned(tuned_rows),
        "generator_variants": summarise_variants(variant_rows),
        "joint_sensitivity": summarise_joint(joint_rows),
    }
    (HERE / "extensions_results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(joint_rows).to_csv(HERE / "joint_sensitivity.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("bins_")} for r in variant_rows]).to_csv(
        HERE / "generator_variants.csv", index=False)
    print(json.dumps({k: results[k] for k in ("calibration", "generator_variants", "joint_sensitivity")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
