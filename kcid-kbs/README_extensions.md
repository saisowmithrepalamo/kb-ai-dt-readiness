# KCID-AI: additional robustness analyses

`extensions.py` adds four analyses for the revised manuscript. It imports
`analysis.py` and `sensitivity.py` from this folder unchanged, and before running
it checks that its generator reproduces the published seed-23 run exactly.

| Analysis | Design | Output |
|---|---|---|
| Calibration | Brier score, calibration-in-the-large and slope, expected calibration error over deciles, error against the true probability; seed 23 and seeds 1–50 | `extensions_results.json` (`calibration`), `Figure_4.png/.pdf` |
| Cross-validated boosting | 5-fold grid search on each training sample (log loss; 100/250/500 trees, learning rate 0.02/0.05/0.1, depth 2/3), with and without the rules; seeds 1–50 | `extensions_results.json` (`tuned_boosting`) |
| Generator structures | Published generator plus five variants (control gaps ×0.5 and ×1.5, no venture gap, linear controls, no control signal); seeds 1–50 each | `extensions_results.json` (`generator_variants`), `generator_variants.csv` |
| Joint sensitivity | 500 joint draws of the six payoff and policy constants over the one-at-a-time ranges (seed 2026), on the seed-23 fits | `extensions_results.json` (`joint_sensitivity`), `joint_sensitivity.csv` |

## Run

Use the pinned versions in `requirements.txt`, from this folder:

```bash
export MPLCONFIGDIR="$(pwd)/.mplconfig"
python extensions.py   # about 35 minutes on 8 cores
```

`analysis.py` and `sensitivity.py` do not need to be run first; `extensions.py`
reads the committed `sensitivity_results.json` only for its reproduction check.
