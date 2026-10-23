# Knowledge-based AI decision support for digital transformation

This repository holds the code, data outputs and manuscripts for two related papers. Both use reproducible **synthetic** data; no real firm or breach records are included.

| Folder | Paper | Status |
|---|---|---|
| [`kcid-kbs/`](kcid-kbs/) | **KCID-AI**: cybersecurity-aware screening of digital transformation initiatives | **Current.** Full-length research article prepared for *Knowledge-Based Systems* |
| repository root | Readiness score from enterprise innovation data | Earlier short communication prepared for *Journal of Innovation & Knowledge* |

---

## Current paper: KCID-AI (`kcid-kbs/`)

**A knowledge-based AI decision-support system for cybersecurity-aware digital transformation screening using innovation and entrepreneurship indicators**

### Read the paper

| File | What it is |
|---|---|
| [`KCID-AI_manuscript.pdf`](kcid-kbs/KCID-AI_manuscript.pdf) | The manuscript (20 double-spaced pages) |
| [`paper.docx`](kcid-kbs/paper.docx) | Word version for submission |
| [`paper.md`](kcid-kbs/paper.md) | Editable source text |
| [`supplementary_material.md`](kcid-kbs/supplementary_material.md) / [`.docx`](kcid-kbs/supplementary_material.docx) | Table S1, Note S1 and Fig. S1 |
| [`highlights.txt`](kcid-kbs/highlights.txt) | Five highlights for the submission system |

Author name, affiliation and email are left blank in the manuscript files.

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

### Reproduce every number

Requires Python 3 (versions are pinned in `kcid-kbs/requirements.txt`).

```bash
cd kcid-kbs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MPLCONFIGDIR="$(pwd)/.mplconfig"
python analysis.py
python sensitivity.py
```

Run `analysis.py` first; `sensitivity.py` imports it. `analysis.py` takes about 20 seconds and `sensitivity.py` about 2–3 minutes. With the pinned versions, both scripts reproduce the committed JSON, CSV and PNG files exactly. Afterwards, `git status` lists only the three PDF figures, which differ only in their embedded creation date.

| Script | Writes | Used in the paper for |
|---|---|---|
| `analysis.py` | `results.json`, `table1_descriptives.csv`, `table2_performance.csv`, `Figure_1.png/.pdf`, `Figure_2.png/.pdf` | Figs. 1–2, the first six rows of Table 1, Supplementary Table S1 |
| `sensitivity.py` | `sensitivity_results.json`, `payoff_sensitivity.csv`, `Figure_3.png/.pdf` | The remaining rows of Table 1, Table 2, Supplementary Note S1 and Fig. S1 |

Fixed settings: population seed 23, bootstrap seed 24 with 1,000 resamples, replication seeds 1–50.

### Other files

`kcid-kbs/review/` contains notes from an earlier editing pass of the short-communication version of this paper. They are kept for the record and are superseded by the current manuscript.

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
