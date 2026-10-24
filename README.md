# Knowledge-based AI decision support for digital transformation

This repository holds the code and data outputs for two related papers, and the manuscript of the earlier one. Both use reproducible **synthetic** data; no real firm or breach records are included.

| Folder | Paper | Status |
|---|---|---|
| [`kcid-kbs/`](kcid-kbs/) | **KCID-AI**: cybersecurity-aware screening of digital transformation initiatives | **Current.** Full-length research article prepared for *Knowledge-Based Systems* |
| repository root | Readiness score from enterprise innovation data | Earlier short communication prepared for *Journal of Innovation & Knowledge* |

---

## Current paper: KCID-AI (`kcid-kbs/`)

**Breach-risk models and security rules for screening digital transformation initiatives**

The manuscript is not posted here while it is under consideration at a journal. The folder holds the code, the simulated dataset and every result file; [`kcid-kbs/README.md`](kcid-kbs/README.md) describes each file, the dataset's columns, which script produces which table and figure, and how to reproduce them.

### What the study does

KCID-AI recommends **launch**, **harden** or **defer** for each digital transformation initiative. It has two separately switchable parts:

1. a **logistic breach-probability score** built from five opportunity indicators (innovation, entrepreneurship, data volume, data sensitivity, transformation breadth) and six cybersecurity indicators (access control, encryption, monitoring, incident response, skills, vendor control);
2. a **propositional rule layer**: three hard rules force deferral, and four soft rules subtract a penalty from the screening score.

The paper compares KCID-AI with eleven other policies on a synthetic population of 5,000 initiatives (1,251 held out). It repeats the comparison over 50 regenerated populations and across a grid of payoff and threshold settings.

### Main findings

- Adding the cybersecurity indicators raises decision utility in all 50 replications.
- The rules reduce launched breaches in every replication, but they do not raise utility.
- An expected-utility policy capped at KCID-AI's expected-breach budget launches about as many breaches as KCID-AI, with higher utility in 46 of 50 replications.
- A hybrid of KCID-AI's hard rules and that capped policy matches the capped policy's utility. In every replication, though, it explains fewer non-launch decisions with a named rule and launches more initiatives that carry a flagged control gap.
- In short, the rule layer buys rule-consistent, auditable decisions, not a better utility–risk trade-off. A rule-only policy without the probability score is worse on both utility and launched breaches.

### Reproduce

From `kcid-kbs/`, with the versions pinned in `kcid-kbs/requirements.txt`:

```bash
cd kcid-kbs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MPLCONFIGDIR="$(pwd)/.mplconfig"
python analysis.py
python sensitivity.py
python extensions.py
python export_dataset.py
```

Fixed settings: population seed 23, bootstrap seed 24 with 1,000 resamples, replication seeds 1–50.

---

## Earlier paper: readiness score (repository root)

**A knowledge-based AI decision-support model for predicting digital transformation readiness from enterprise innovation data**

Prepared as a short communication for *Journal of Innovation & Knowledge*. The paper is [`paper.md`](paper.md). The enterprises are synthetic, and `analysis.py` is the data-generating process, the model, and the source of every number in the paper.

### What the model does

1. Builds an innovation index from R&D, patents, product innovation, process innovation and collaboration.
2. Keeps data intensity separate.
3. Builds a knowledge index from codification, sharing, training and digital skills.
4. Scores readiness with a logistic model of those indexes and the data × knowledge product.
5. Compares that score with a logistic model and a gradient-boosting model that see only the six innovation and data indicators.

### Reproduce

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MPLCONFIGDIR="$(pwd)/.mplconfig"
python analysis.py
```

`analysis.py` rewrites `figure1_architecture.png`, `figure2_knowledge_effect.png`, `table1_descriptives.csv`, `table2_performance.csv` and `results.json`. The seed is 23. Training uses 3,750 enterprises, and the reported metrics use the remaining 1,250.
