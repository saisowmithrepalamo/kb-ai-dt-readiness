#!/usr/bin/env python3
"""Reproducible illustration for the short communication.

A knowledge-based AI decision-support model for predicting digital
transformation readiness from enterprise innovation data.

The enterprises are synthetic. The data-generating process is the hypothesis
stated in the paper: data intensity raises readiness only when knowledge is
codified. Running this file rewrites the two figures, the two tables, and
results.json. Do not edit those outputs by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 23
N = 5000
TEST_SIZE = 0.25
TARGET_PREVALENCE = 0.40
BOOTSTRAP = 1000

INNOVATION = [
    "rd_intensity",
    "log_patents",
    "product_innovation",
    "process_innovation",
    "collaboration_breadth",
    "data_intensity",
]
KNOWLEDGE = [
    "knowledge_codification",
    "knowledge_sharing",
    "training_intensity",
    "digital_skills",
]
INTERACTION = "data_x_codification"
LABEL = "dt_ready"

LABELS = {
    "rd_intensity": "R&D intensity",
    "log_patents": "Patent stock (log)",
    "product_innovation": "Product innovation",
    "process_innovation": "Process innovation",
    "collaboration_breadth": "Collaboration breadth",
    "data_intensity": "Data intensity",
    "knowledge_codification": "Knowledge codification",
    "knowledge_sharing": "Knowledge sharing",
    "training_intensity": "Training intensity",
    "digital_skills": "Digital skills",
    INTERACTION: "Data × codification",
}

INK = "#1c2834"
BLUE = "#1d4e89"
TEAL = "#0e6b67"
CLAY = "#b5523a"
SAND = "#f7f4ee"
PAPER = "#fffdf9"
GRID = "#d9d3c7"


def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def _latent_factors(rng: np.random.Generator, n: int) -> np.ndarray:
    """Four correlated firm factors: innovation, knowledge, data, skills."""
    corr = np.array(
        [
            [1.00, 0.15, 0.18, 0.12],
            [0.15, 1.00, 0.10, 0.32],
            [0.18, 0.10, 1.00, 0.12],
            [0.12, 0.32, 0.12, 1.00],
        ]
    )
    z = rng.multivariate_normal(np.zeros(4), corr, size=n)
    return 1.0 / (1.0 + np.exp(-0.95 * z))


def simulate(rng: np.random.Generator, n: int, intercept: float) -> pd.DataFrame:
    innovation, knowledge, data, skills = _latent_factors(rng, n).T
    shock = rng.normal(0.0, 1.0, size=n)

    frame = pd.DataFrame(
        {
            "rd_intensity": _clip(0.015 + 0.18 * innovation + rng.normal(0, 0.010, n), 0, 0.40),
            "log_patents": np.maximum(0.10 + 2.5 * innovation + rng.normal(0, 0.18, n), 0),
            "product_innovation": _clip(innovation + rng.normal(0, 0.07, n), 0, 1),
            "process_innovation": _clip(0.88 * innovation + 0.12 * data + rng.normal(0, 0.07, n), 0, 1),
            "collaboration_breadth": _clip(0.82 * innovation + 0.18 * knowledge + rng.normal(0, 0.07, n), 0, 1),
            "data_intensity": _clip(data + rng.normal(0, 0.05, n), 0, 1),
            "knowledge_codification": _clip(knowledge + rng.normal(0, 0.045, n), 0, 1),
            "knowledge_sharing": _clip(0.92 * knowledge + 0.08 * skills + rng.normal(0, 0.05, n), 0, 1),
            "training_intensity": _clip(0.72 * skills + 0.28 * knowledge + rng.normal(0, 0.05, n), 0, 1),
            "digital_skills": _clip(skills + rng.normal(0, 0.05, n), 0, 1),
        }
    )
    # The hypothesis: data raises readiness only through codified knowledge.
    eta = (
        intercept
        + 0.55 * innovation
        + 1.35 * knowledge
        + 0.05 * data
        + 2.60 * data * knowledge
        + 0.45 * skills
        + 0.12 * shock
    )
    propensity = 1.0 / (1.0 + np.exp(-eta))
    frame[LABEL] = rng.binomial(1, propensity)
    frame[INTERACTION] = frame["data_intensity"] * frame["knowledge_codification"]
    return frame


def calibrate_intercept(seed: int) -> float:
    """Choose the intercept so about 40% of enterprises are ready."""
    lo, hi = -4.0, 1.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        sample = simulate(np.random.default_rng(seed), N, mid)
        if sample[LABEL].mean() > TARGET_PREVALENCE:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def fit_logistic(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")),
        ]
    )
    model.fit(x_train, y_train)
    return model


def fit_boosting(x_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.85,
        random_state=SEED,
    )
    model.fit(x_train, y_train)
    return model


def slice_mask(frame: pd.DataFrame, data_cut: float) -> pd.Series:
    """Data-rich and knowledge-poor: high data intensity, codification below 0.40."""
    return (frame["data_intensity"] >= data_cut) & (frame["knowledge_codification"] < 0.40)


def false_positive_rate(y_true: np.ndarray, proba: np.ndarray, mask: np.ndarray) -> float:
    pred = (proba >= 0.5).astype(int)
    negatives = mask & (y_true == 0)
    if negatives.sum() == 0:
        return float("nan")
    return float(((pred == 1) & negatives).sum() / negatives.sum())


def metric_row(name: str, features: str, y_true: np.ndarray, proba: np.ndarray, mask: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "features": features,
        "auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "fpr_data_rich_knowledge_poor": false_positive_rate(y_true, proba, mask),
        "n_slice": int(mask.sum()),
    }


def bootstrap_auc(y_true: np.ndarray, proba: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    scores = []
    n = len(y_true)
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = y_true[idx], proba[idx]
        if len(np.unique(y_b)) < 2:
            continue
        scores.append(roc_auc_score(y_b, p_b))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_auc_difference(
    y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, rng: np.random.Generator
) -> dict:
    """Paired bootstrap of AUC(a) minus AUC(b) on the same test enterprises."""
    diffs = []
    n = len(y_true)
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        diffs.append(roc_auc_score(y_b, proba_a[idx]) - roc_auc_score(y_b, proba_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"difference": float(np.mean(diffs)), "ci_low": float(lo), "ci_high": float(hi)}


def descriptives(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in INNOVATION + KNOWLEDGE:
        role = "Innovation and data" if column in INNOVATION else "Knowledge process"
        rows.append(
            {
                "indicator": LABELS[column],
                "role": role,
                "mean": frame[column].mean(),
                "sd": frame[column].std(ddof=1),
                "p10": frame[column].quantile(0.10),
                "p90": frame[column].quantile(0.90),
                "corr_ready": frame[column].corr(frame[LABEL]),
            }
        )
    return pd.DataFrame(rows)


def performance_table(rows: list[dict], intervals: dict[str, tuple[float, float]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    table["auc_ci_low"] = table["model"].map(lambda name: intervals[name][0])
    table["auc_ci_high"] = table["model"].map(lambda name: intervals[name][1])
    return table


class KnowledgeDecisionModel:
    """Transparent score: innovation index, data, knowledge index, and their product.

    Indexes are standardized on the training sample. The data coefficient is
    the effect of data when the knowledge index is at its training mean.
    """

    NAMES = ["Innovation index", "Data intensity", "Knowledge index", "Data × knowledge"]

    def fit(self, frame: pd.DataFrame, y: pd.Series) -> "KnowledgeDecisionModel":
        self.activity = [column for column in INNOVATION if column != "data_intensity"]
        raw = self._raw_indexes(frame)
        self.center = raw.mean(axis=0)
        self.scale = raw.std(axis=0, ddof=0)
        self.scale[self.scale == 0] = 1.0
        self.clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        self.clf.fit(self._design(frame), y)
        return self

    def _z(self, frame: pd.DataFrame, column: str) -> np.ndarray:
        values = frame[column].to_numpy(dtype=float)
        return (values - self.item_mean[column]) / self.item_sd[column]

    def _raw_indexes(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "item_mean"):
            columns = self.activity + ["data_intensity"] + KNOWLEDGE
            self.item_mean = frame[columns].mean()
            self.item_sd = frame[columns].std(ddof=0).replace(0, 1.0)
        innovation = np.mean([self._z(frame, column) for column in self.activity], axis=0)
        data = self._z(frame, "data_intensity")
        knowledge = np.mean([self._z(frame, column) for column in KNOWLEDGE], axis=0)
        return np.column_stack([innovation, data, knowledge])

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        scaled = (self._raw_indexes(frame) - self.center) / self.scale
        interaction = scaled[:, 1] * scaled[:, 2]
        return np.column_stack([scaled, interaction])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._design(frame))[:, 1]

    def coefficient_table(self) -> pd.DataFrame:
        coef = self.clf.coef_[0]
        out = pd.DataFrame({"feature": self.NAMES, "log_odds": coef, "odds_ratio": np.exp(coef)})
        out.loc[len(out)] = ["Intercept", self.clf.intercept_[0], np.exp(self.clf.intercept_[0])]
        return out


def quintile_curves(test: pd.DataFrame) -> pd.DataFrame:
    plot_df = test.copy()
    plot_df["quintile"] = pd.qcut(
        plot_df["knowledge_codification"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )
    median_data = plot_df["data_intensity"].median()
    plot_df["data_band"] = np.where(
        plot_df["data_intensity"] >= median_data, "High data intensity", "Low data intensity"
    )
    rows = []
    for (band, quintile), part in plot_df.groupby(["data_band", "quintile"], observed=True):
        n = len(part)
        observed = part[LABEL].mean()
        rows.append(
            {
                "data_band": band,
                "quintile": str(quintile),
                "n": n,
                "observed": observed,
                "observed_se": float(np.sqrt(observed * (1 - observed) / n)),
                "knowledge_model": part["p_knowledge"].mean(),
                "data_only_model": part["p_data_only"].mean(),
            }
        )
    return pd.DataFrame(rows)


def draw_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    boxes = [
        (0.35, 0.85, 3.15, 3.15, BLUE, "1  Enterprise innovation data",
         "R&D intensity\nPatent stock\nProduct innovation\nProcess innovation\nCollaboration breadth\nData intensity"),
        (4.05, 0.85, 3.15, 3.15, TEAL, "2  Knowledge layer",
         "Codification\nKnowledge sharing\nTraining intensity\nDigital skills\nData × knowledge index"),
        (7.75, 0.85, 3.10, 3.15, CLAY, "3  AI decision support",
         "Standardized logistic score\nReadiness probability\nEarly / emerging / ready\nAction: scale, pair,\nor build knowledge first"),
    ]
    for x, y, w, h, color, title, body in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.012,rounding_size=0.08",
                facecolor=SAND, edgecolor=color, linewidth=1.8,
            )
        )
        ax.text(x + 0.16, y + h - 0.28, title, ha="left", va="top", fontsize=11, color=color, fontweight="bold")
        ax.text(x + 0.18, y + h - 0.62, body, ha="left", va="top", fontsize=9.2, color=INK, linespacing=1.45)

    for start, end in ((3.50, 4.05), (7.20, 7.75)):
        ax.add_patch(
            FancyArrowPatch(
                (start, 2.42), (end, 2.42),
                arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=INK,
            )
        )
    ax.text(5.6, 0.38, "Readiness is scored only after data passes through the knowledge layer.",
            ha="center", va="center", fontsize=9, color=INK, style="italic")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def draw_knowledge_effect(curves: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    fig.patch.set_facecolor(PAPER)
    order = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    x = np.arange(len(order))
    for ax, band in zip(axes, ["Low data intensity", "High data intensity"]):
        part = curves[curves["data_band"] == band].set_index("quintile").loc[order]
        ax.set_facecolor(PAPER)
        ax.errorbar(
            x, part["observed"], yerr=1.96 * part["observed_se"],
            fmt="o", color=INK, ms=6, lw=1.2, capsize=3, label="Observed readiness", zorder=3,
        )
        ax.plot(x, part["knowledge_model"], color=TEAL, lw=2.2, marker="s", ms=5, label="Knowledge-based model")
        ax.plot(x, part["data_only_model"], color=CLAY, lw=1.8, ls="--", marker="^", ms=5, label="Data-only logistic")
        ax.set_xticks(x)
        ax.set_xticklabels(["Lowest\nQ1", "Q2", "Q3", "Q4", "Highest\nQ5"])
        ax.set_title(band, color=INK, fontsize=12, pad=8)
        ax.set_xlabel("Knowledge codification")
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#b7b1a6")
        ax.spines["bottom"].set_color("#b7b1a6")
    axes[0].set_ylabel("P(digital transformation ready)")
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_tables(desc: pd.DataFrame, perf: pd.DataFrame) -> None:
    desc_out = desc.copy()
    for column in ("mean", "sd", "p10", "p90", "corr_ready"):
        desc_out[column] = desc_out[column].map(lambda v: f"{v:.3f}")
    desc_out.to_csv(ROOT / "table1_descriptives.csv", index=False)

    perf_out = perf.copy()
    perf_out["auc_95ci"] = perf_out.apply(
        lambda r: f"{r['auc']:.3f} ({r['auc_ci_low']:.3f}–{r['auc_ci_high']:.3f})", axis=1
    )
    show = perf_out[
        ["model", "features", "auc_95ci", "brier", "balanced_accuracy", "fpr_data_rich_knowledge_poor", "n_slice"]
    ].copy()
    for column in ("brier", "balanced_accuracy", "fpr_data_rich_knowledge_poor"):
        show[column] = show[column].map(lambda v: f"{v:.3f}")
    show.to_csv(ROOT / "table2_performance.csv", index=False)


def main() -> None:
    rng = np.random.default_rng(SEED)
    intercept = calibrate_intercept(SEED)
    population = simulate(rng, N, intercept)
    train, test = train_test_split(
        population, test_size=TEST_SIZE, random_state=SEED, stratify=population[LABEL]
    )
    data_cut = float(train["data_intensity"].median())
    y_test = test[LABEL].to_numpy()
    mask = slice_mask(test, data_cut).to_numpy()

    data_logit = fit_logistic(train[INNOVATION], train[LABEL])
    data_boost = fit_boosting(train[INNOVATION], train[LABEL])
    knowledge_model = KnowledgeDecisionModel().fit(train, train[LABEL])

    p_data = data_logit.predict_proba(test[INNOVATION])[:, 1]
    p_boost = data_boost.predict_proba(test[INNOVATION])[:, 1]
    p_knowledge = knowledge_model.predict_proba(test)

    feature_note = {
        "Data-only logistic": "Six innovation and data indicators",
        "Data-only gradient boosting": "Same six indicators, black-box AI",
        "Knowledge-based logistic": "Innovation index, data, knowledge index, and data × knowledge",
    }
    probs = {
        "Data-only logistic": p_data,
        "Data-only gradient boosting": p_boost,
        "Knowledge-based logistic": p_knowledge,
    }
    rows = [
        metric_row(name, feature_note[name], y_test, proba, mask)
        for name, proba in probs.items()
    ]
    boot_rng = np.random.default_rng(SEED + 7)
    intervals = {name: bootstrap_auc(y_test, proba, boot_rng) for name, proba in probs.items()}
    auc_gaps = {
        "knowledge_minus_logistic": bootstrap_auc_difference(y_test, p_knowledge, p_data, boot_rng),
        "knowledge_minus_boosting": bootstrap_auc_difference(y_test, p_knowledge, p_boost, boot_rng),
    }

    desc = descriptives(population)
    perf = performance_table(rows, intervals)
    coef = knowledge_model.coefficient_table()
    named = dict(zip(coef["feature"], coef["log_odds"]))
    if named["Knowledge index"] <= 0 or named["Data × knowledge"] <= 0:
        raise RuntimeError(f"Knowledge-layer coefficients are not positive: {named}")

    test_plot = test.copy()
    test_plot["p_knowledge"] = p_knowledge
    test_plot["p_data_only"] = p_data
    curves = quintile_curves(test_plot)

    draw_architecture(ROOT / "figure1_architecture.png")
    draw_knowledge_effect(curves, ROOT / "figure2_knowledge_effect.png")
    write_tables(desc, perf)

    knowledge_auc = rows[2]["auc"]
    data_auc = max(rows[0]["auc"], rows[1]["auc"])
    if not np.isfinite(knowledge_auc) or knowledge_auc < data_auc + 0.04:
        raise RuntimeError(
            f"Knowledge AUC {knowledge_auc:.3f} is not at least 0.04 above the best data-only AUC {data_auc:.3f}."
        )
    if not (0.28 <= population[LABEL].mean() <= 0.52):
        raise RuntimeError(f"Prevalence {population[LABEL].mean():.3f} is outside the target band.")

    payload = {
        "seed": SEED,
        "n": N,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "intercept": intercept,
        "prevalence": float(population[LABEL].mean()),
        "test_prevalence": float(test[LABEL].mean()),
        "data_intensity_train_median": data_cut,
        "codification_cutoff": 0.40,
        "decision_threshold": 0.5,
        "descriptives": desc.to_dict(orient="records"),
        "performance": perf.to_dict(orient="records"),
        "auc_gaps": auc_gaps,
        "coefficients": coef.to_dict(orient="records"),
        "curves": curves.to_dict(orient="records"),
        "slice_ready_rate": float(test.loc[slice_mask(test, data_cut), LABEL].mean()),
        "high_data_ready_by_codification_tail": {
            "low_codification_observed": float(
                test.loc[
                    (test["data_intensity"] >= test["data_intensity"].median())
                    & (test["knowledge_codification"] <= test["knowledge_codification"].quantile(0.2)),
                    LABEL,
                ].mean()
            ),
            "high_codification_observed": float(
                test.loc[
                    (test["data_intensity"] >= test["data_intensity"].median())
                    & (test["knowledge_codification"] >= test["knowledge_codification"].quantile(0.8)),
                    LABEL,
                ].mean()
            ),
        },
    }
    (ROOT / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"prevalence": payload["prevalence"], "performance": payload["performance"], "coefficients": payload["coefficients"]}, indent=2))


if __name__ == "__main__":
    main()
