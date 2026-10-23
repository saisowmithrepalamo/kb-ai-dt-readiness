# Knowledge-based AI decision support for digital transformation readiness

The latest manuscript is the *Knowledge-Based Systems* full-length research article in `kcid-kbs/`.

**A knowledge-based AI decision-support system for cybersecurity-aware digital transformation screening using innovation and entrepreneurship indicators**

Open `kcid-kbs/KCID-AI_manuscript.pdf`. The source is `kcid-kbs/paper.md`, and the supplementary material is `kcid-kbs/supplementary_material.md`.

- `kcid-kbs/analysis.py` rebuilds Figures 1–2, the first six rows of Table 1, Supplementary Table S1 and `results.json`.
- `kcid-kbs/sensitivity.py` imports `analysis.py` and rebuilds the remaining rows of Table 1, Table 2 (50 regenerated populations), the payoff grid (`payoff_sensitivity.csv`), Supplementary Fig. S1 (`Figure_3.png`) and `sensitivity_results.json`. It also evaluates the boosting-plus-rules, expected-utility, capped expected-utility, rule-only and hybrid policies. Runtime is about 2–3 minutes.

The earlier readiness-score package remains in this directory.

## Earlier package

Replication package for the short communication:

**A knowledge-based AI decision-support model for predicting digital transformation readiness from enterprise innovation data**

Target journal: *Journal of Innovation & Knowledge*.

The paper is `paper.md`. The enterprises are synthetic. The script is the data-generating process, the model, and the source of every number in the paper.

## What the model does

1. Builds an innovation index from R&D, patents, product innovation, process innovation, and collaboration.
2. Keeps data intensity separate.
3. Builds a knowledge index from codification, sharing, training, and digital skills.
4. Scores readiness with a logistic model of those indexes and the data × knowledge product.
5. Compares that score with a logistic model and a gradient-boosting model that see only the six innovation and data indicators.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MPLCONFIGDIR="$(pwd)/.mplconfig"
python analysis.py
```

`analysis.py` rewrites:

- `figure1_architecture.png`
- `figure2_knowledge_effect.png`
- `table1_descriptives.csv`
- `table2_performance.csv`
- `results.json`

Seed is 23. Training uses 3,750 enterprises and the reported metrics use the remaining 1,250.
