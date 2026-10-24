"""KCID-AI: knowledge-based cybersecurity decision support.

Reproducible synthetic benchmark for a Knowledge-Based Systems short
communication. Seed 23. Every number in paper.md is read from results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 23
N = 5000
TRAIN_FRAC = 0.75
N_BOOT = 1000

DATA_COLS = ["inn", "ent", "data_vol", "data_sens", "dt"]
KNOW_COLS = ["access", "crypto", "monitor", "respond", "skills", "vendor"]
ALL_COLS = DATA_COLS + KNOW_COLS

LAUNCH, HARDEN, DEFER = 2, 1, 0
ACTION_NAME = {LAUNCH: "launch", HARDEN: "harden", DEFER: "defer"}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate(rng: np.random.Generator) -> pd.DataFrame:
    """Latent breach process. Rules are not the label."""
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
    # Controls are only weakly tied to innovation, so a data-rich initiative
    # can still be control-poor.
    access = _sigmoid(0.95 * z_ctrl + 0.10 * z_innov + 0.25 * rng.normal(size=N))
    crypto = _sigmoid(0.80 * z_ctrl + 0.25 * rng.normal(size=N))
    monitor = _sigmoid(0.85 * z_ctrl + 0.05 * z_data + 0.30 * rng.normal(size=N))
    respond = _sigmoid(0.75 * z_ctrl + 0.28 * rng.normal(size=N))
    skills = _sigmoid(0.50 * z_ctrl + 0.20 * z_innov + 0.40 * rng.normal(size=N))
    vendor = _sigmoid(0.65 * z_ctrl + 0.30 * rng.normal(size=N))

    gap_access = np.clip(data_sens - access, 0, None)
    gap_mon = np.clip(data_vol - monitor, 0, None)
    gap_crypto = np.clip(data_sens - crypto, 0, None) * (data_sens > 0.75)
    venture_gap = ((ent > 0.70) & (skills < 0.45) & (data_sens > 0.50)).astype(float)
    noise = rng.normal(size=N)

    # Intercept is set so the mean latent breach probability is 0.36.
    # The slope terms stay fixed. This is a design calibration, not a fit to outcomes.
    def mean_p(b0: float) -> float:
        eta = (
            b0
            + 0.35 * data_sens
            + 0.25 * data_vol
            + 0.15 * dt
            + 0.10 * ent
            - 0.10 * access
            - 0.08 * monitor
            + 3.10 * gap_access
            + 2.15 * gap_mon
            + 1.80 * gap_crypto
            + 1.35 * venture_gap
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
        + 0.35 * data_sens
        + 0.25 * data_vol
        + 0.15 * dt
        + 0.10 * ent
        - 0.10 * access
        - 0.08 * monitor
        + 3.10 * gap_access
        + 2.15 * gap_mon
        + 1.80 * gap_crypto
        + 1.35 * venture_gap
        + 0.25 * noise
    )
    p_true = _sigmoid(eta)
    breach = rng.random(N) < p_true
    impact = 25.0 + 90.0 * data_sens + 40.0 * data_vol + 20.0 * dt
    value = 18.0 + 35.0 * inn + 22.0 * ent + 12.0 * dt

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
            "impact": impact,
            "value": value,
            "venture_gap": venture_gap.astype(int),
        }
    )


def knowledge_rules(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Forward-chaining propositional rules. Hard rules exclude a launch.

    Thresholds are stated before any test metric is computed.
    """
    hard = (
        ((df["data_sens"] > 0.70) & (df["access"] < 0.45))
        | ((df["data_sens"] > 0.82) & (df["crypto"] < 0.40))
        | ((df["data_vol"] > 0.72) & (df["monitor"] < 0.38) & (df["data_sens"] > 0.50))
    )
    soft = np.zeros(len(df), dtype=int)
    soft += ((df["data_vol"] > 0.65) & (df["monitor"] < 0.42)).to_numpy().astype(int)
    soft += ((df["dt"] > 0.60) & (df["respond"] < 0.40)).to_numpy().astype(int)
    soft += (
        (df["ent"] > 0.70) & (df["skills"] < 0.45) & (df["data_sens"] > 0.50)
    ).to_numpy().astype(int)
    soft += ((df["data_sens"] > 0.60) & (df["vendor"] < 0.40)).to_numpy().astype(int)
    return hard.to_numpy(), soft


def decide(
    p: np.ndarray,
    hard: np.ndarray,
    soft: np.ndarray,
    value: np.ndarray,
    impact: np.ndarray,
    use_rules: bool,
) -> np.ndarray:
    """Launch when innovation value exceeds expected breach loss.

    The probability source and the rule switch are the only differences
    across models. A hard rule forces defer. Each soft rule subtracts 8
    from the score. Thresholds are fixed from the objective, not from the
    test metrics.
    """
    penalty = (soft * 8.0) if use_rules else np.zeros(len(p))
    score = value - p * impact - penalty
    action = np.full(len(p), DEFER, dtype=int)
    action[score >= -5.0] = HARDEN
    action[score >= 10.0] = LAUNCH
    if use_rules:
        action[hard] = DEFER
    return action


def utility(action: np.ndarray, breach: np.ndarray, value: np.ndarray, impact: np.ndarray) -> np.ndarray:
    u = np.zeros(len(action))
    launch = action == LAUNCH
    harden = action == HARDEN
    u[launch & (breach == 0)] = value[launch & (breach == 0)]
    u[launch & (breach == 1)] = value[launch & (breach == 1)] - impact[launch & (breach == 1)]
    u[harden & (breach == 0)] = 0.55 * value[harden & (breach == 0)] - 5.0
    u[harden & (breach == 1)] = 0.55 * value[harden & (breach == 1)] - 5.0 - 0.35 * impact[harden & (breach == 1)]
    return u


def fit_proba(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], kind: str) -> np.ndarray:
    y = train["breach"].to_numpy()
    x_tr = train[cols]
    x_te = test[cols]
    if kind == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
        )
    elif kind == "boost":
        model = GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
            random_state=SEED,
        )
    else:
        raise ValueError(kind)
    model.fit(x_tr, y)
    return model.predict_proba(x_te)[:, 1]


def summarize(action: np.ndarray, breach: np.ndarray, u: np.ndarray, cell: np.ndarray) -> dict:
    launch = action == LAUNCH
    breached = breach == 1
    n_breach = int(breached.sum())
    n_launch = int(launch.sum())
    cell_breach = cell & breached
    return {
        "n": int(len(action)),
        "launch_share": float(launch.mean()),
        "harden_share": float((action == HARDEN).mean()),
        "defer_share": float((action == DEFER).mean()),
        "missed_breach_rate": float(launch[breached].mean()) if n_breach else None,
        "breach_among_launches": float(breached[launch].mean()) if n_launch else None,
        "mean_utility": float(u.mean()),
        "cell_n": int(cell.sum()),
        "cell_breaches": int(cell_breach.sum()),
        "cell_missed_breach_rate": float(launch[cell_breach].mean()) if cell_breach.sum() else None,
        "cell_launches": int((cell & launch).sum()),
        "cell_breach_among_launches": float(breached[cell & launch].mean()) if (cell & launch).sum() else None,
    }


def auc_of(p: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", direction="out", length=3.5, width=0.8, labelsize=9)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def _save_figures(risk_rows: list[dict], launch_rows: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    observed = "#222222"
    data_c = "#D55E00"
    know_c = "#0072B2"
    kcid_c = "#009E73"
    xs = [r["access_quintile"] for r in risk_rows]
    labels = ["Q1\nlowest access", "Q2", "Q3", "Q4", "Q5\nhighest access"]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    y = np.array([r["observed_rate"] for r in risk_rows])
    lo = y - np.array([r["observed_lo"] for r in risk_rows])
    hi = np.array([r["observed_hi"] for r in risk_rows]) - y
    ax.errorbar(
        xs, y, yerr=[lo, hi], fmt="o", color=observed, ms=6.5, lw=1.2, capsize=3, capthick=1.0, zorder=3, label="Observed breach rate"
    )
    ax.plot(xs, [r["p_data_logistic"] for r in risk_rows], "s--", color=data_c, ms=6, lw=1.8, label="Data-only score")
    ax.plot(xs, [r["p_know_logistic"] for r in risk_rows], "o-", color=know_c, ms=6, lw=1.8, label="Knowledge score")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Access-control quintile among data-rich held-out initiatives")
    ax.set_ylabel("Breach probability")
    _style(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ROOT / "Figure_1.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(ROOT / "Figure_1.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    launch_labels = [
        f"Q1\nlowest access\n{launch_rows[0]['n_breach']} breaches",
        f"Q2\n{launch_rows[1]['n_breach']} breaches",
        f"Q3\n{launch_rows[2]['n_breach']} breaches",
        f"Q4\n{launch_rows[3]['n_breach']} breaches",
        f"Q5\nhighest access\n{launch_rows[4]['n_breach']} breaches",
    ]
    ax.plot(xs, [r["data_logistic_miss"] for r in launch_rows], "s--", color=data_c, ms=6.5, lw=1.8, label="Data-only logistic")
    ax.plot(xs, [r["know_logistic_miss"] for r in launch_rows], "o-", color=know_c, ms=6.5, lw=1.8, label="Knowledge logistic, rules off")
    ax.plot(xs, [r["kcid_miss"] for r in launch_rows], "D-", color=kcid_c, ms=6, lw=1.8, label="KCID-AI")
    ax.set_xticks(xs)
    ax.set_xticklabels(launch_labels)
    ax.set_xlabel("Access-control quintile among data-rich held-out initiatives")
    ax.set_ylabel("Share of breaches that are launched")
    _style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ROOT / "Figure_2.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(ROOT / "Figure_2.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = generate(rng)
    hard, soft = knowledge_rules(df)
    df["hard"] = hard.astype(int)
    df["soft"] = soft

    idx = np.arange(N)
    rng_split = np.random.default_rng(SEED)
    rng_split.shuffle(idx)
    # Stratify by breach with a deterministic cut of the shuffled index.
    breach_idx = idx[df.loc[idx, "breach"].to_numpy() == 1]
    safe_idx = idx[df.loc[idx, "breach"].to_numpy() == 0]
    n_tr_b = int(TRAIN_FRAC * len(breach_idx))
    n_tr_s = int(TRAIN_FRAC * len(safe_idx))
    train_idx = np.concatenate([breach_idx[:n_tr_b], safe_idx[:n_tr_s]])
    test_idx = np.concatenate([breach_idx[n_tr_b:], safe_idx[n_tr_s:]])
    rng_split.shuffle(train_idx)
    rng_split.shuffle(test_idx)

    train = df.loc[train_idx].reset_index(drop=True)
    test = df.loc[test_idx].reset_index(drop=True)
    # Data-rich and control-poor: the cell in which innovation and data
    # indicators are least able to stand in for cybersecurity knowledge.
    vol_cut = float(train["data_vol"].median())
    access_cut = float(np.quantile(train["access"], 0.40))
    cell = (test["data_vol"].to_numpy() >= vol_cut) & (test["access"].to_numpy() <= access_cut)

    specs = {
        "data_logistic": ("logistic", DATA_COLS, False),
        "data_boost": ("boost", DATA_COLS, False),
        "know_logistic": ("logistic", ALL_COLS, False),
        "know_boost": ("boost", ALL_COLS, False),
        "data_logistic_rules": ("logistic", DATA_COLS, True),
        "kcid": ("logistic", ALL_COLS, True),
    }

    probs = {}
    actions = {}
    metrics = {}
    y = test["breach"].to_numpy()
    for name, (kind, cols, use_rules) in specs.items():
        p = fit_proba(train, test, cols, kind)
        a = decide(
            p,
            test["hard"].to_numpy().astype(bool),
            test["soft"].to_numpy(),
            test["value"].to_numpy(),
            test["impact"].to_numpy(),
            use_rules,
        )
        u = utility(a, y, test["value"].to_numpy(), test["impact"].to_numpy())
        probs[name] = p
        actions[name] = a
        row = summarize(a, y, u, cell)
        row["auc"] = auc_of(p, y)
        metrics[name] = row

    # Paired bootstrap of the test set for the contrasts the paper reports.
    boot_rng = np.random.default_rng(SEED + 1)
    contrasts = {
        "kcid_minus_data_logistic": ("kcid", "data_logistic"),
        "kcid_minus_know_logistic": ("kcid", "know_logistic"),
        "kcid_minus_know_boost": ("kcid", "know_boost"),
        "know_logistic_minus_data_logistic": ("know_logistic", "data_logistic"),
    }
    diffs = {k: {"miss": [], "utility": [], "cell_miss": []} for k in contrasts}
    n_te = len(test)
    for _ in range(N_BOOT):
        draw = boot_rng.integers(0, n_te, n_te)
        y_b = y[draw]
        cell_b = cell[draw]
        for key, (left, right) in contrasts.items():
            u_l = utility(actions[left][draw], y_b, test["value"].to_numpy()[draw], test["impact"].to_numpy()[draw])
            u_r = utility(actions[right][draw], y_b, test["value"].to_numpy()[draw], test["impact"].to_numpy()[draw])
            s_l = summarize(actions[left][draw], y_b, u_l, cell_b)
            s_r = summarize(actions[right][draw], y_b, u_r, cell_b)
            diffs[key]["miss"].append(s_l["missed_breach_rate"] - s_r["missed_breach_rate"])
            diffs[key]["utility"].append(s_l["mean_utility"] - s_r["mean_utility"])
            diffs[key]["cell_miss"].append(
                (s_l["cell_missed_breach_rate"] or 0.0) - (s_r["cell_missed_breach_rate"] or 0.0)
            )

    def pack(samples: list[float]) -> dict:
        arr = np.asarray(samples, dtype=float)
        # Two-sided bootstrap p against a zero difference.
        p_lo = float(np.mean(arr <= 0))
        p_hi = float(np.mean(arr >= 0))
        return {
            "mean": float(arr.mean()),
            "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
            "p_twosided": float(min(1.0, 2 * min(p_lo, p_hi))),
        }

    contrast_out = {
        key: {metric: pack(samples) for metric, samples in bucket.items()}
        for key, bucket in diffs.items()
    }

    # Descriptives on the full population.
    corr_rows = []
    for col in ALL_COLS:
        corr_rows.append(
            {
                "indicator": col,
                "mean": float(df[col].mean()),
                "sd": float(df[col].std(ddof=1)),
                "p10": float(np.quantile(df[col], 0.10)),
                "p90": float(np.quantile(df[col], 0.90)),
                "corr_breach": float(np.corrcoef(df[col], df["breach"])[0, 1]),
            }
        )

    # Access quintiles inside the data-rich half of the test set.
    # Figure 1 uses every data-rich initiative. Figure 2 uses the breaches inside it.
    data_rich = test["data_vol"].to_numpy() >= vol_cut
    access_te = test["access"].to_numpy()
    qs = np.quantile(access_te[data_rich], [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(access_te, qs)

    def wilson(k: int, n: int) -> tuple[float, float]:
        z = 1.96
        phat = k / n
        den = 1.0 + z * z / n
        centre = (phat + z * z / (2.0 * n)) / den
        half = z * np.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / den
        return float(centre - half), float(centre + half)

    risk_rows = []
    launch_rows = []
    for q in range(5):
        mask_all = data_rich & (bins == q)
        n_all = int(mask_all.sum())
        k_breach = int(y[mask_all].sum())
        lo, hi = wilson(k_breach, n_all)
        risk_rows.append(
            {
                "access_quintile": q + 1,
                "n": n_all,
                "n_breach": k_breach,
                "observed_rate": k_breach / n_all,
                "observed_lo": lo,
                "observed_hi": hi,
                "p_data_logistic": float(probs["data_logistic"][mask_all].mean()),
                "p_know_logistic": float(probs["know_logistic"][mask_all].mean()),
            }
        )
        mask_b = mask_all & (y == 1)
        launch_rows.append(
            {
                "access_quintile": q + 1,
                "n_breach": int(mask_b.sum()),
                "data_logistic_miss": float((actions["data_logistic"][mask_b] == LAUNCH).mean()),
                "know_logistic_miss": float((actions["know_logistic"][mask_b] == LAUNCH).mean()),
                "kcid_miss": float((actions["kcid"][mask_b] == LAUNCH).mean()),
            }
        )

    separation = {
        "n_data_rich": int(data_rich.sum()),
        "corr_access_breach": float(np.corrcoef(access_te[data_rich], y[data_rich])[0, 1]),
        "corr_access_p_data": float(np.corrcoef(access_te[data_rich], probs["data_logistic"][data_rich])[0, 1]),
        "corr_access_p_know": float(np.corrcoef(access_te[data_rich], probs["know_logistic"][data_rich])[0, 1]),
    }

    results = {
        "seed": SEED,
        "n": N,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "breach_rate": float(df["breach"].mean()),
        "mean_p_true": float(df["p_true"].mean()),
        "hard_rate": float(df["hard"].mean()),
        "vol_cut": vol_cut,
        "access_cut": access_cut,
        "descriptives": corr_rows,
        "metrics": metrics,
        "contrasts": contrast_out,
        "figure_risk": risk_rows,
        "figure_launch": launch_rows,
        "separation": separation,
        "logistic_coefs": {},
    }

    # Standardized logistic coefficients on the training sample, for the text.
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(train[ALL_COLS])
    logit = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    logit.fit(x_tr, train["breach"])
    results["logistic_coefs"] = {
        "intercept": float(logit.intercept_[0]),
        "coef": {col: float(c) for col, c in zip(ALL_COLS, logit.coef_[0])},
    }

    (ROOT / "results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(corr_rows).to_csv(ROOT / "table1_descriptives.csv", index=False)
    metric_rows = []
    for name, row in metrics.items():
        metric_rows.append({"model": name, **row})
    pd.DataFrame(metric_rows).to_csv(ROOT / "table2_performance.csv", index=False)

    _save_figures(risk_rows, launch_rows)

    print(json.dumps({"breach_rate": results["breach_rate"], "metrics": metrics, "contrasts": contrast_out}, indent=2))


if __name__ == "__main__":
    main()
