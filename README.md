# Knowledge-based AI decision support for digital transformation readiness

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
