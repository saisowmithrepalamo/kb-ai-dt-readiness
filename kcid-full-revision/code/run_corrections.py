"""Corrected KCID-AI experiments.

The design below was fixed before any corrected test result was inspected.
It does not retune rules, payoffs, or the alternative generator after the run.

Protocols
---------
published
    Probability models fit on the full training split. The capped policy's
    price lambda is bisected on the test portfolio. This is the published
    procedure, re-executed here.

train_freeze
    Same fitted probabilities as ``published``. Lambda is bisected on
    in-sample training predictions and then frozen. The test set is not used
    to choose lambda. Training predictions are not a held-out sample.

corrected
    The training split is divided, before any test metric, into a fit sample
    (80%) and a validation sample (20%), stratified on the breach label.
    Models are selected and fit on the fit sample only. A Platt calibrator
    is fit on the validation sample only. Lambda is bisected on the
    validation sample only, then frozen and applied to the test set.

Alternative generator
    Same indicators as the published generator. The breach equation drops the
    clipped control-gap terms and the venture-gap indicator that repeats
    rule r6. It is still a simulation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import analysis_published as published  # noqa: E402

ROOT = CODE.parent
OUT = ROOT / "results_new"
OUT.mkdir(parents=True, exist_ok=True)

LAUNCH, HARDEN, DEFER = published.LAUNCH, published.HARDEN, published.DEFER
DATA_COLS = published.DATA_COLS
ALL_COLS = published.ALL_COLS
N = published.N
TRAIN_FRAC = published.TRAIN_FRAC

# Locked before the run. Not searched.
VAL_FRAC = 0.20
VAL_SEED_OFFSET = 7919
PLATT_C = 1.0e6
LOGISTIC_C_GRID = (0.1, 1.0, 10.0)
GB_GRID = (
    {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.85},
    {"n_estimators": 100, "learning_rate": 0.10, "max_depth": 2, "subsample": 0.85},
)
MISSPEC = {"retain": 0.40, "h_cost": 10.0, "resid": 0.55}
TRUE_PAY = {"retain": 0.55, "h_cost": 5.0, "resid": 0.35}
N_BINS = 10
N_BOOT = 1000


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def generate_alternative(rng: np.random.Generator) -> pd.DataFrame:
    """Same indicators as ``published.generate``. Different breach equation.

    The equation has no clipped gap and no venture-gap indicator.
    Coefficients were written down before the run. The intercept is set so
    the mean latent probability is 0.36, as in the published generator.
    """
    z_innov = rng.normal(size=N)
    z_ent = rng.normal(size=N)
    z_data = rng.normal(size=N)
    z_ctrl = rng.normal(size=N)
    z_dt = 0.40 * z_innov + 0.30 * z_data + rng.normal(scale=0.85, size=N)
    inn = _sigmoid(0.85 * z_innov + 0.20 * rng.normal(size=N))
    ent = _sigmoid(0.80 * z_ent + 0.15 * z_innov + 0.20 * rng.normal(size=N))
    data_vol = _sigmoid(0.90 * z_data + 0.15 * rng.normal(size=N))
    data_sens = _sigmoid(0.45 * z_data + 0.85 * rng.normal(size=N))
    dt = _sigmoid(z_dt)
    access = _sigmoid(0.95 * z_ctrl + 0.10 * z_innov + 0.25 * rng.normal(size=N))
    crypto = _sigmoid(0.80 * z_ctrl + 0.25 * rng.normal(size=N))
    monitor = _sigmoid(0.85 * z_ctrl + 0.05 * z_data + 0.30 * rng.normal(size=N))
    respond = _sigmoid(0.75 * z_ctrl + 0.28 * rng.normal(size=N))
    skills = _sigmoid(0.50 * z_ctrl + 0.20 * z_innov + 0.40 * rng.normal(size=N))
    vendor = _sigmoid(0.65 * z_ctrl + 0.30 * rng.normal(size=N))
    noise = rng.normal(size=N)

    def mean_p(b0: float) -> float:
        eta = (
            b0
            + 0.55 * data_sens
            + 0.40 * data_vol
            + 0.20 * dt
            + 0.15 * ent
            - 0.25 * access
            - 0.15 * monitor
            - 0.10 * crypto
            + 1.20 * data_sens * (1.0 - access)
            + 0.80 * data_vol * (1.0 - monitor)
            + 0.50 * data_sens * (1.0 - crypto)
            + 0.25 * noise
        )
        return float(_sigmoid(eta).mean())

    lo, hi = -8.0, 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if mean_p(mid) > 0.36:
            hi = mid
        else:
            lo = mid
    b0 = 0.5 * (lo + hi)
    eta = (
        b0
        + 0.55 * data_sens
        + 0.40 * data_vol
        + 0.20 * dt
        + 0.15 * ent
        - 0.25 * access
        - 0.15 * monitor
        - 0.10 * crypto
        + 1.20 * data_sens * (1.0 - access)
        + 0.80 * data_vol * (1.0 - monitor)
        + 0.50 * data_sens * (1.0 - crypto)
        + 0.25 * noise
    )
    p_true = _sigmoid(eta)
    breach = rng.random(N) < p_true
    venture_gap = ((ent > 0.70) & (skills < 0.45) & (data_sens > 0.50)).astype(int)
    return pd.DataFrame(
        {
            "inn": inn,
            "ent": ent,
            "data_vol": data_vol,
            "data_sens": data_sens,
            "dt": dt,
            "access": access,
            "crypto": crypto,
            "monitor": monitor,
            "respond": respond,
            "skills": skills,
            "vendor": vendor,
            "p_true": p_true,
            "breach": breach.astype(int),
            "impact": 25.0 + 90.0 * data_sens + 40.0 * data_vol + 20.0 * dt,
            "value": 18.0 + 35.0 * inn + 22.0 * ent + 12.0 * dt,
            "venture_gap": venture_gap,
            "intercept": b0,
        }
    )


def stratified_split(df: pd.DataFrame, seed: int, frac: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    y = df["breach"].to_numpy()
    breach_idx = idx[y[idx] == 1]
    safe_idx = idx[y[idx] == 0]
    n_b = int(frac * len(breach_idx))
    n_s = int(frac * len(safe_idx))
    left = np.concatenate([breach_idx[:n_b], safe_idx[:n_s]])
    right = np.concatenate([breach_idx[n_b:], safe_idx[n_s:]])
    rng.shuffle(left)
    rng.shuffle(right)
    return left, right


def population_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match ``analysis_published.main``: shuffle with the population seed."""
    train_idx, test_idx = stratified_split(df, seed, TRAIN_FRAC)
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    return train, test


def cell_mask(reference: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    vol_cut = float(reference["data_vol"].median())
    access_cut = float(np.quantile(reference["access"], 0.40))
    mask = (target["data_vol"].to_numpy() >= vol_cut) & (target["access"].to_numpy() <= access_cut)
    return mask, vol_cut, access_cut


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if not np.any(mask):
            continue
        total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def safe_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))


def calibration_summary(y: np.ndarray, p: np.ndarray) -> dict:
    slope_model = LogisticRegression(C=PLATT_C, solver="lbfgs", max_iter=500)
    slope_model.fit(logit(p).reshape(-1, 1), y)
    return {
        "brier": float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "log_loss": safe_log_loss(y, p),
        "ece_10": ece_score(y, p),
        "cal_intercept": float(slope_model.intercept_[0]),
        "cal_slope": float(slope_model.coef_[0, 0]),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
    }


def fit_platt(p: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    if len(np.unique(y)) < 2:
        return None
    model = LogisticRegression(C=PLATT_C, solver="lbfgs", max_iter=500)
    model.fit(logit(p).reshape(-1, 1), y)
    return model


def apply_platt(model: LogisticRegression | None, p: np.ndarray) -> np.ndarray:
    if model is None:
        return p
    return model.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def select_logistic_c(x: pd.DataFrame, y: np.ndarray, seed: int) -> dict:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = []
    best_c, best_loss = 1.0, np.inf
    for c_value in LOGISTIC_C_GRID:
        fold_losses = []
        for train_idx, valid_idx in splitter.split(x, y):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c_value, max_iter=500, solver="lbfgs"),
            )
            model.fit(x.iloc[train_idx], y[train_idx])
            prob = model.predict_proba(x.iloc[valid_idx])[:, 1]
            fold_losses.append(safe_log_loss(y[valid_idx], prob))
        mean_loss = float(np.mean(fold_losses))
        scores.append({"C": c_value, "mean_log_loss": mean_loss})
        if mean_loss < best_loss:
            best_c, best_loss = c_value, mean_loss
    return {"selected_C": best_c, "cv_log_loss": best_loss, "grid": scores}


def select_gb(x: pd.DataFrame, y: np.ndarray, seed: int) -> dict:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = []
    best, best_loss = GB_GRID[0], np.inf
    for params in GB_GRID:
        fold_losses = []
        for train_idx, valid_idx in splitter.split(x, y):
            model = GradientBoostingClassifier(random_state=seed, **params)
            model.fit(x.iloc[train_idx], y[train_idx])
            prob = model.predict_proba(x.iloc[valid_idx])[:, 1]
            fold_losses.append(safe_log_loss(y[valid_idx], prob))
        mean_loss = float(np.mean(fold_losses))
        row = dict(params)
        row["mean_log_loss"] = mean_loss
        scores.append(row)
        if mean_loss < best_loss:
            best, best_loss = params, mean_loss
    return {"selected": best, "cv_log_loss": best_loss, "grid": scores}


def fit_logistic(train: pd.DataFrame, cols: list[str], c_value: float) -> object:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=500, solver="lbfgs"),
    )
    model.fit(train[cols], train["breach"].to_numpy())
    return model


def fit_gb(train: pd.DataFrame, cols: list[str], params: dict, seed: int) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(random_state=seed, **params)
    model.fit(train[cols], train["breach"].to_numpy())
    return model


def predict(model, frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return model.predict_proba(frame[cols])[:, 1]


def decide_threshold(p, hard, soft, value, impact, use_rules: bool, penalty: float = 8.0) -> np.ndarray:
    pen = soft * penalty if use_rules else np.zeros(len(p))
    score = value - p * impact - pen
    action = np.full(len(p), DEFER, dtype=int)
    action[score >= -5.0] = HARDEN
    action[score >= 10.0] = LAUNCH
    if use_rules:
        action[hard] = DEFER
    return action


def decide_eu(p, hard, value, impact, use_hard: bool, pay: dict, lam: float = 0.0) -> np.ndarray:
    e_launch = value - p * impact - lam * p
    e_harden = pay["retain"] * value - pay["h_cost"] - pay["resid"] * p * impact
    action = np.full(len(p), DEFER, dtype=int)
    best = np.zeros(len(p))
    action[e_harden > best] = HARDEN
    best = np.maximum(best, e_harden)
    action[e_launch > best] = LAUNCH
    if use_hard:
        action[hard] = DEFER
    return action


def choose_lambda(p, hard, value, impact, pay: dict, budget: float, use_hard: bool) -> tuple[np.ndarray, float]:
    def expected_launches(lam: float) -> tuple[float, np.ndarray]:
        action = decide_eu(p, hard, value, impact, use_hard, pay, lam)
        return float(p[action == LAUNCH].sum()), action

    unconstrained, action0 = expected_launches(0.0)
    if unconstrained <= budget:
        return action0, 0.0
    lo, hi = 0.0, 1.0
    while expected_launches(hi)[0] > budget:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if expected_launches(mid)[0] > budget:
            lo = mid
        else:
            hi = mid
    action = expected_launches(hi)[1]
    return action, float(hi)


def decide_rules_only(hard, soft) -> np.ndarray:
    action = np.full(len(hard), LAUNCH, dtype=int)
    action[soft > 0] = HARDEN
    action[hard] = DEFER
    return action


def realised_utility(action, breach, value, impact, pay: dict = TRUE_PAY) -> np.ndarray:
    utility = np.zeros(len(action))
    launch = action == LAUNCH
    harden = action == HARDEN
    utility[launch] = value[launch] - breach[launch] * impact[launch]
    utility[harden] = (
        pay["retain"] * value[harden] - pay["h_cost"] - pay["resid"] * breach[harden] * impact[harden]
    )
    return utility


def score_policy(action, frame: pd.DataFrame, cell: np.ndarray, p_for_auc: np.ndarray | None) -> dict:
    y = frame["breach"].to_numpy()
    value = frame["value"].to_numpy()
    impact = frame["impact"].to_numpy()
    utility = realised_utility(action, y, value, impact)
    launch = action == LAUNCH
    hard = frame["hard"].to_numpy().astype(bool)
    soft = frame["soft"].to_numpy()
    named = hard | (soft > 0)
    n_breach = int((y == 1).sum())
    cell_breach = cell & (y == 1)
    row = {
        "utility": float(utility.mean()),
        "launch_share": float(launch.mean()),
        "launched_breaches": int((launch & (y == 1)).sum()),
        "lb_rate": float((launch & (y == 1)).sum() / n_breach) if n_breach else None,
        "cell_launched_breaches": int((launch & (y == 1) & cell).sum()),
        "cell_lb_rate": float((launch[cell_breach]).mean()) if cell_breach.any() else None,
        "reason_coverage": float(named[~launch].mean()) if (~launch).any() else None,
        "flagged_launch_share": float((soft[launch] > 0).mean()) if launch.any() else None,
        "n": int(len(action)),
        "n_breach": n_breach,
        "cell_breaches": int(cell_breach.sum()),
    }
    if p_for_auc is not None and len(np.unique(y)) > 1:
        row["auc"] = float(roc_auc_score(y, p_for_auc))
    return row


def budget_of(action, p) -> float:
    return float(p[action == LAUNCH].sum())


def run_published(train, test, cell, seed: int) -> dict:
    """Published hyperparameters and test-set lambda."""
    published.SEED = seed
    y_tr = train["breach"].to_numpy()
    data_model = fit_logistic(train, DATA_COLS, 1.0)
    know_model = fit_logistic(train, ALL_COLS, 1.0)
    boost_model = fit_gb(train, ALL_COLS, GB_GRID[0], seed)
    p_data = predict(data_model, test, DATA_COLS)
    p_know = predict(know_model, test, ALL_COLS)
    p_boost = predict(boost_model, test, ALL_COLS)
    p_data_tr = predict(data_model, train, DATA_COLS)
    p_know_tr = predict(know_model, train, ALL_COLS)
    hard = test["hard"].to_numpy().astype(bool)
    soft = test["soft"].to_numpy()
    value = test["value"].to_numpy()
    impact = test["impact"].to_numpy()
    hard_tr = train["hard"].to_numpy().astype(bool)
    value_tr = train["value"].to_numpy()
    impact_tr = train["impact"].to_numpy()
    actions = {
        "data_logistic": decide_threshold(p_data, hard, soft, value, impact, False),
        "know_logistic": decide_threshold(p_know, hard, soft, value, impact, False),
        "know_boost": decide_threshold(p_boost, hard, soft, value, impact, False),
        "data_logistic_rules": decide_threshold(p_data, hard, soft, value, impact, True),
        "kcid": decide_threshold(p_know, hard, soft, value, impact, True),
        "rules_only": decide_rules_only(hard, soft),
    }
    actions["eu"] = decide_eu(p_know, hard, value, impact, False, TRUE_PAY, 0.0)
    actions["eu_hard"] = decide_eu(p_know, hard, value, impact, True, TRUE_PAY, 0.0)
    kcid_tr = decide_threshold(
        p_know_tr,
        hard_tr,
        train["soft"].to_numpy(),
        value_tr,
        impact_tr,
        True,
    )
    train_budget = budget_of(kcid_tr, p_know_tr)
    # Lambda is chosen on training rows only, then applied unchanged to the test set.
    lam_train_only = choose_lambda(
        p_know_tr, hard_tr, value_tr, impact_tr, TRUE_PAY, train_budget, False
    )[1]
    actions["eu_capped_trainfreeze"] = decide_eu(
        p_know, hard, value, impact, False, TRUE_PAY, lam_train_only
    )
    actions["hybrid_trainfreeze"] = decide_eu(
        p_know, hard, value, impact, True, TRUE_PAY, lam_train_only
    )
    test_budget = budget_of(actions["kcid"], p_know)
    actions["eu_capped_leaky"], lam_leaky = choose_lambda(
        p_know, hard, value, impact, TRUE_PAY, test_budget, False
    )
    actions["hybrid_leaky"], _ = choose_lambda(
        p_know, hard, value, impact, TRUE_PAY, test_budget, True
    )
    actions["eu_misspec"] = decide_eu(p_know, hard, value, impact, False, MISSPEC, 0.0)
    probs = {
        "data_logistic": p_data,
        "know_logistic": p_know,
        "know_boost": p_boost,
        "data_logistic_rules": p_data,
        "kcid": p_know,
        "rules_only": None,
        "eu": p_know,
        "eu_hard": p_know,
        "eu_capped_trainfreeze": p_know,
        "hybrid_trainfreeze": p_know,
        "eu_capped_leaky": p_know,
        "hybrid_leaky": p_know,
        "eu_misspec": p_know,
    }
    metrics = {name: score_policy(action, test, cell, probs[name]) for name, action in actions.items()}
    metrics["_lambda_train"] = lam_train_only
    metrics["_lambda_leaky"] = lam_leaky
    metrics["_train_budget"] = train_budget
    metrics["_test_budget"] = test_budget
    metrics["_actions"] = actions
    metrics["_p_know"] = p_know
    metrics["_p_data_train_unused"] = p_data_tr
    return metrics


def run_corrected(train, test, seed: int, do_gb_search: bool) -> dict:
    fit_idx, val_idx = stratified_split(train, seed + VAL_SEED_OFFSET, 1.0 - VAL_FRAC)
    fit = train.iloc[fit_idx].reset_index(drop=True)
    valid = train.iloc[val_idx].reset_index(drop=True)
    y_fit = fit["breach"].to_numpy()
    y_val = valid["breach"].to_numpy()
    c_info = select_logistic_c(fit[ALL_COLS], y_fit, seed)
    c_data = select_logistic_c(fit[DATA_COLS], y_fit, seed)
    if do_gb_search:
        gb_info = select_gb(fit[ALL_COLS], y_fit, seed)
        gb_params = gb_info["selected"]
    else:
        gb_info = {"selected": GB_GRID[0], "cv_log_loss": None, "grid": [], "skipped": True}
        gb_params = GB_GRID[0]
    data_model = fit_logistic(fit, DATA_COLS, c_data["selected_C"])
    know_model = fit_logistic(fit, ALL_COLS, c_info["selected_C"])
    boost_model = fit_gb(fit, ALL_COLS, gb_params, seed)
    raw = {
        "data": (
            predict(data_model, valid, DATA_COLS),
            predict(data_model, test, DATA_COLS),
        ),
        "know": (
            predict(know_model, valid, ALL_COLS),
            predict(know_model, test, ALL_COLS),
        ),
        "boost": (
            predict(boost_model, valid, ALL_COLS),
            predict(boost_model, test, ALL_COLS),
        ),
    }
    platt = {
        "data": fit_platt(raw["data"][0], y_val),
        "know": fit_platt(raw["know"][0], y_val),
        "boost": fit_platt(raw["boost"][0], y_val),
    }
    cal_val = {name: apply_platt(platt[name], raw[name][0]) for name in raw}
    cal_te = {name: apply_platt(platt[name], raw[name][1]) for name in raw}
    cell, vol_cut, access_cut = cell_mask(fit, test)
    hard = test["hard"].to_numpy().astype(bool)
    soft = test["soft"].to_numpy()
    value = test["value"].to_numpy()
    impact = test["impact"].to_numpy()
    hard_v = valid["hard"].to_numpy().astype(bool)
    soft_v = valid["soft"].to_numpy()
    value_v = valid["value"].to_numpy()
    impact_v = valid["impact"].to_numpy()
    kcid_val = decide_threshold(cal_val["know"], hard_v, soft_v, value_v, impact_v, True)
    val_budget = budget_of(kcid_val, cal_val["know"])
    _, lam = choose_lambda(cal_val["know"], hard_v, value_v, impact_v, TRUE_PAY, val_budget, False)
    p_know = cal_te["know"]
    p_data = cal_te["data"]
    p_boost = cal_te["boost"]
    actions = {
        "data_logistic": decide_threshold(p_data, hard, soft, value, impact, False),
        "know_logistic": decide_threshold(p_know, hard, soft, value, impact, False),
        "know_boost": decide_threshold(p_boost, hard, soft, value, impact, False),
        "data_logistic_rules": decide_threshold(p_data, hard, soft, value, impact, True),
        "kcid": decide_threshold(p_know, hard, soft, value, impact, True),
        "kcid_raw": decide_threshold(raw["know"][1], hard, soft, value, impact, True),
        "rules_only": decide_rules_only(hard, soft),
        "eu": decide_eu(p_know, hard, value, impact, False, TRUE_PAY, 0.0),
        "eu_capped": decide_eu(p_know, hard, value, impact, False, TRUE_PAY, lam),
        "hybrid": decide_eu(p_know, hard, value, impact, True, TRUE_PAY, lam),
        "eu_misspec": decide_eu(p_know, hard, value, impact, False, MISSPEC, 0.0),
    }
    probs = {name: p_know for name in actions}
    probs["data_logistic"] = p_data
    probs["data_logistic_rules"] = p_data
    probs["know_boost"] = p_boost
    probs["rules_only"] = None
    probs["kcid_raw"] = raw["know"][1]
    metrics = {name: score_policy(action, test, cell, probs[name]) for name, action in actions.items()}
    metrics["_lambda"] = lam
    metrics["_val_budget"] = val_budget
    metrics["_n_fit"] = int(len(fit))
    metrics["_n_val"] = int(len(valid))
    metrics["_vol_cut"] = vol_cut
    metrics["_access_cut"] = access_cut
    metrics["_logistic_c"] = c_info
    metrics["_data_c"] = c_data
    metrics["_gb"] = gb_info
    metrics["_cal_test_know"] = calibration_summary(test["breach"].to_numpy(), p_know)
    metrics["_raw_test_know"] = calibration_summary(test["breach"].to_numpy(), raw["know"][1])
    metrics["_cal_test_boost"] = calibration_summary(test["breach"].to_numpy(), p_boost)
    metrics["_raw_test_boost"] = calibration_summary(test["breach"].to_numpy(), raw["boost"][1])
    metrics["_actions"] = actions
    metrics["_p_know_cal"] = p_know
    metrics["_p_know_raw"] = raw["know"][1]
    return metrics


def flat_metrics(prefix: str, metrics: dict, names: list[str]) -> dict:
    row = {}
    for name in names:
        block = metrics[name]
        for key, value in block.items():
            if key.startswith("_"):
                continue
            row[f"{prefix}{name}.{key}"] = value
    return row


def one_seed(seed: int, generator: str, do_gb_search: bool) -> dict:
    rng = np.random.default_rng(seed)
    if generator == "original":
        frame = published.generate(rng)
    elif generator == "alternative":
        frame = generate_alternative(rng)
    else:
        raise ValueError(generator)
    hard, soft = published.knowledge_rules(frame)
    frame = frame.copy()
    frame["hard"] = hard.astype(int)
    frame["soft"] = soft
    train, test = population_split(frame, seed)
    cell, _, _ = cell_mask(train, test)
    published_metrics = run_published(train, test, cell, seed)
    corrected = run_corrected(train, test, seed, do_gb_search=do_gb_search)
    pub_names = [
        "data_logistic",
        "know_logistic",
        "know_boost",
        "data_logistic_rules",
        "kcid",
        "rules_only",
        "eu",
        "eu_hard",
        "eu_capped_leaky",
        "hybrid_leaky",
        "eu_capped_trainfreeze",
        "hybrid_trainfreeze",
        "eu_misspec",
    ]
    cor_names = [
        "data_logistic",
        "know_logistic",
        "know_boost",
        "data_logistic_rules",
        "kcid",
        "kcid_raw",
        "rules_only",
        "eu",
        "eu_capped",
        "hybrid",
        "eu_misspec",
    ]
    row = {"seed": seed, "generator": generator}
    row.update(flat_metrics("pub_", published_metrics, pub_names))
    row.update(flat_metrics("cor_", corrected, cor_names))
    row["pub_lambda_leaky"] = published_metrics["_lambda_leaky"]
    row["pub_lambda_train"] = published_metrics["_lambda_train"]
    row["cor_lambda"] = corrected["_lambda"]
    row["cor_selected_C"] = corrected["_logistic_c"]["selected_C"]
    row["cor_selected_C_data"] = corrected["_data_c"]["selected_C"]
    row["cor_gb_n"] = corrected["_gb"]["selected"]["n_estimators"]
    row["cor_gb_lr"] = corrected["_gb"]["selected"]["learning_rate"]
    row["cor_gb_depth"] = corrected["_gb"]["selected"]["max_depth"]
    row["cor_raw_ece"] = corrected["_raw_test_know"]["ece_10"]
    row["cor_cal_ece"] = corrected["_cal_test_know"]["ece_10"]
    row["cor_raw_brier"] = corrected["_raw_test_know"]["brier"]
    row["cor_cal_brier"] = corrected["_cal_test_know"]["brier"]
    row["cor_raw_logloss"] = corrected["_raw_test_know"]["log_loss"]
    row["cor_cal_logloss"] = corrected["_cal_test_know"]["log_loss"]
    row["cor_raw_slope"] = corrected["_raw_test_know"]["cal_slope"]
    row["cor_cal_slope"] = corrected["_cal_test_know"]["cal_slope"]
    row["cor_raw_intercept"] = corrected["_raw_test_know"]["cal_intercept"]
    row["cor_cal_intercept"] = corrected["_cal_test_know"]["cal_intercept"]
    row["cor_boost_raw_ece"] = corrected["_raw_test_boost"]["ece_10"]
    row["cor_boost_cal_ece"] = corrected["_cal_test_boost"]["ece_10"]
    row["cor_boost_raw_brier"] = corrected["_raw_test_boost"]["brier"]
    row["cor_boost_cal_brier"] = corrected["_cal_test_boost"]["brier"]
    leaky = published_metrics["_actions"]["eu_capped_leaky"] == LAUNCH
    hybrid = published_metrics["_actions"]["hybrid_leaky"] == LAUNCH
    row["pub_launch_set_equal"] = bool(np.array_equal(leaky, hybrid))
    row["pub_harden_to_defer"] = int(
        (
            (published_metrics["_actions"]["eu_capped_leaky"] == HARDEN)
            & (published_metrics["_actions"]["hybrid_leaky"] == DEFER)
        ).sum()
    )
    row["cor_launch_set_equal"] = bool(
        np.array_equal(
            corrected["_actions"]["eu_capped"] == LAUNCH,
            corrected["_actions"]["hybrid"] == LAUNCH,
        )
    )
    row["cor_harden_to_defer"] = int(
        (
            (corrected["_actions"]["eu_capped"] == HARDEN)
            & (corrected["_actions"]["hybrid"] == DEFER)
        ).sum()
    )
    detail = {
        "published_actions": published_metrics["_actions"],
        "corrected_actions": corrected["_actions"],
        "published_p": published_metrics["_p_know"],
        "corrected_p": corrected["_p_know_cal"],
        "test_index_frame": test,
        "cell": cell,
        "cv": {
            "logistic": corrected["_logistic_c"],
            "data_logistic": corrected["_data_c"],
            "gb": corrected["_gb"],
        },
        "calibration_know_raw": corrected["_raw_test_know"],
        "calibration_know_cal": corrected["_cal_test_know"],
    }
    return row, detail


def bootstrap_contrasts(detail: dict, seed: int = 24) -> dict:
    frame = detail["test_index_frame"]
    cell = detail["cell"]
    y = frame["breach"].to_numpy()
    value = frame["value"].to_numpy()
    impact = frame["impact"].to_numpy()
    pairs = {
        "pub_rules_minus_data": ("published_actions", "data_logistic_rules", "data_logistic"),
        "pub_kcid_minus_data": ("published_actions", "kcid", "data_logistic"),
        "pub_kcid_minus_leaky": ("published_actions", "kcid", "eu_capped_leaky"),
        "pub_kcid_minus_trainfreeze": ("published_actions", "kcid", "eu_capped_trainfreeze"),
        "cor_kcid_minus_capped": ("corrected_actions", "kcid", "eu_capped"),
        "cor_kcid_minus_raw": ("corrected_actions", "kcid", "kcid_raw"),
        "cor_cal_know_minus_data": ("corrected_actions", "know_logistic", "data_logistic"),
    }
    rng = np.random.default_rng(seed)
    n = len(frame)
    store = {key: [] for key in pairs}
    for _ in range(N_BOOT):
        draw = rng.integers(0, n, n)
        y_b = y[draw]
        cell_b = cell[draw]
        for key, (bucket, left, right) in pairs.items():
            actions = detail[bucket]
            u_l = realised_utility(actions[left][draw], y_b, value[draw], impact[draw]).mean()
            u_r = realised_utility(actions[right][draw], y_b, value[draw], impact[draw]).mean()
            store[key].append(float(u_l - u_r))
    out = {}
    for key, samples in store.items():
        arr = np.asarray(samples)
        out[key] = {
            "mean": float(arr.mean()),
            "p2_5": float(np.quantile(arr, 0.025)),
            "p97_5": float(np.quantile(arr, 0.975)),
        }
    return out


def summarise(frame: pd.DataFrame) -> dict:
    numeric = frame.select_dtypes(include=[np.number])
    out = {}
    for column in numeric.columns:
        if column == "seed":
            continue
        values = numeric[column].dropna().to_numpy()
        if len(values) == 0:
            continue
        out[column] = {
            "mean": float(values.mean()),
            "p2_5": float(np.quantile(values, 0.025)),
            "p97_5": float(np.quantile(values, 0.975)),
            "n_pos": int((values > 1e-12).sum()),
            "n_neg": int((values < -1e-12).sum()),
            "n_zero": int(len(values) - (values > 1e-12).sum() - (values < -1e-12).sum()),
            "n": int(len(values)),
        }
    return out


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        left, right = text.split("-")
        return list(range(int(left), int(right) + 1))
    return [int(part) for part in text.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="23")
    parser.add_argument("--generators", default="original,alternative")
    parser.add_argument("--gb-search", action="store_true")
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    generators = [part.strip() for part in args.generators.split(",") if part.strip()]
    rows = []
    started = time.time()
    seed23_detail = None
    for generator in generators:
        for seed in seeds:
            tick = time.time()
            row, detail = one_seed(seed, generator, do_gb_search=args.gb_search)
            rows.append(row)
            print(
                f"{generator} seed {seed} in {time.time() - tick:.1f}s "
                f"pub_kcid={row['pub_kcid.utility']:.3f} "
                f"cor_kcid={row['cor_kcid.utility']:.3f} "
                f"cor_capped={row['cor_eu_capped.utility']:.3f}",
                flush=True,
            )
            if seed == 23 and generator == "original":
                seed23_detail = detail
                boot = bootstrap_contrasts(detail)
                (OUT / f"{args.tag}_seed23_bootstrap.json").write_text(json.dumps(boot, indent=2))
                # Reliability curve uses test labels only as a diagnostic.
                y = detail["test_index_frame"]["breach"].to_numpy()
                prob_true, prob_pred = calibration_curve(
                    y, detail["corrected_p"], n_bins=8, strategy="quantile"
                )
                raw_true, raw_pred = calibration_curve(
                    y, detail["published_p"], n_bins=8, strategy="quantile"
                )
                (OUT / f"{args.tag}_seed23_reliability.json").write_text(
                    json.dumps(
                        {
                            "corrected_calibrated": {
                                "fraction_positive": prob_true.tolist(),
                                "mean_predicted": prob_pred.tolist(),
                            },
                            "published_raw": {
                                "fraction_positive": raw_true.tolist(),
                                "mean_predicted": raw_pred.tolist(),
                            },
                            "cv": detail["cv"],
                            "calibration_know_raw": detail["calibration_know_raw"],
                            "calibration_know_cal": detail["calibration_know_cal"],
                        },
                        indent=2,
                    )
                )
                # Drop large arrays before any accidental serialisation.
                del detail
    table = pd.DataFrame(rows)
    csv_path = OUT / f"{args.tag}_replications.csv"
    table.to_csv(csv_path, index=False)
    summary = {
        "elapsed_sec": time.time() - started,
        "seeds": seeds,
        "generators": generators,
        "gb_search": args.gb_search,
        "by_generator": {},
    }
    for generator, part in table.groupby("generator"):
        summary["by_generator"][generator] = summarise(part)
    (OUT / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {csv_path} in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
