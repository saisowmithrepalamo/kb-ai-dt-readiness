# KCID-AI: code and data

Code, simulated data and result files for the article **Breach-risk models and security rules for screening digital transformation initiatives** (Sai Sowmith Repala), prepared for *Knowledge-Based Systems*.

All data here are simulated. No real organisation, person or breach record is included. The manuscript itself is not posted here while it is under consideration at the journal.

## Files

| File | What it is |
|---|---|
| [`kcid_synthetic_population_seed23.csv`](kcid_synthetic_population_seed23.csv) | The simulated dataset behind every seed-23 result: 5,000 initiatives, 1,251 held out (448 breaches). Columns are described below |
| `analysis.py`, `sensitivity.py`, `extensions.py` | The study: data generator, rule layer, models, policies and evaluation |
| `revision.py` | Second revision: the oracle policy that uses the true breach risk, the decomposition of KCID-AI's gap to it, rule layers learned from the data (from breach outcomes and from the model's residuals) compared with the expert rules at matched coverage, and numerical checks of the paper's Propositions 1-3 |
| `export_dataset.py` | Writes the dataset above, using `analysis.py` unchanged |
| `results.json`, `sensitivity_results.json`, `extensions_results.json`, `revision_results.json` | Every number reported in the paper |
| `table1_descriptives.csv`, `table2_performance.csv`, `payoff_sensitivity.csv`, `generator_variants.csv`, `joint_sensitivity.csv`, `revision_seeds.csv` | Result tables written by the scripts |
| `Figure_1`–`Figure_4` (`.png`, `.pdf`) | The scripts' own plots of the results |
| `highlights.txt` | Highlights of the first version of the article |
| [`README_extensions.md`](README_extensions.md) | Details of the four robustness analyses in `extensions.py` |

## Dataset columns

`kcid_synthetic_population_seed23.csv` has one row per simulated digital transformation initiative. Indicators lie on [0, 1].

| Column | Meaning |
|---|---|
| `initiative_id` | Row number, 1–5000 |
| `innovation` | Innovation intensity (opportunity indicator) |
| `entrepreneurship` | Entrepreneurial orientation (opportunity indicator) |
| `data_volume` | Volume of data the initiative uses (opportunity indicator) |
| `data_sensitivity` | Sensitivity of that data (opportunity indicator) |
| `dt_breadth` | Digital-transformation breadth (opportunity indicator) |
| `access_control` | Access-control strength (cybersecurity indicator) |
| `encryption` | Encryption coverage (cybersecurity indicator) |
| `monitoring` | Security monitoring (cybersecurity indicator) |
| `incident_response` | Incident-response readiness (cybersecurity indicator) |
| `security_skills` | Security skills (cybersecurity indicator) |
| `vendor_control` | Third-party (vendor) control (cybersecurity indicator) |
| `true_breach_probability` | The generator's breach probability; known only because the data are simulated, and never used by the models |
| `breach` | 1 if the initiative was breached, drawn from `true_breach_probability` |
| `breach_impact_I` | Breach impact *I*, Eq. (1) of the paper, in illustrative utility units |
| `opportunity_value_V` | Opportunity value *V*, Eq. (2), in illustrative utility units |
| `venture_gap` | 1 if entrepreneurship > 0.70, security skills < 0.45 and data sensitivity > 0.50; a term of the breach process that rule r6 targets |
| `hard_rule` | 1 if a hard rule (r1–r3) fires; hard rules force deferral |
| `soft_rules_fired` | Number of soft rules (r4–r7) that fire; each subtracts a penalty of 8 from the screening score |
| `split` | `train` (75%) or `test` (25%, held out), stratified by breach |

## Which script produces which result

| Script | Writes | Paper |
|---|---|---|
| `analysis.py` | `results.json`, `table1_descriptives.csv`, `table2_performance.csv`, `Figure_1`, `Figure_2` | Table 3 (first six rows), Fig. 2, Supplementary Tables S1 and S2 (first three comparators) |
| `sensitivity.py` | `sensitivity_results.json`, `payoff_sensitivity.csv`, `Figure_3` | Table 3 (remaining rows except the oracle), Table 5, Supplementary Table S2 (remaining comparators) and Fig. S1 |
| `extensions.py` | `extensions_results.json`, `generator_variants.csv`, `joint_sensitivity.csv`, `Figure_4` | Table 6, Supplementary Tables S4 and S5 and Fig. S2 |
| `revision.py` | `revision_results.json`, `revision_seeds.csv` | Table 4, the oracle rows of Tables 3 and 5, the decomposition in Section 4.4, Supplementary Table S6, and the numerical checks of Propositions 1-3 |
| `export_dataset.py` | `kcid_synthetic_population_seed23.csv` | The data behind all seed-23 results |

The expert rules of Table 2 are defined in `knowledge_rules()` in `analysis.py`; Table 1 and Fig. 1 of the paper describe related work and the method, not results. The paper redraws Figs. 2, S1 and S2 at journal size from the same JSON values. Table and figure numbers refer to the second revision of the article and its Supplementary Material.

Fixed settings: population seed 23, bootstrap seed 24 with 1,000 resamples, replication seeds 1–50.

## Reproduce

Requires Python 3; versions are pinned in `requirements.txt`. Run from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MPLCONFIGDIR="$(pwd)/.mplconfig"
python analysis.py         # about 20 seconds
python sensitivity.py      # 2-3 minutes; imports analysis.py
python extensions.py       # about 35 minutes on 8 cores
python revision.py         # about 30 seconds on 8 cores
python export_dataset.py   # a few seconds
```

`extensions.py` and `revision.py` first check that they reproduce the published seed-23 run, and `revision.py` stops if any proposition check fails. With the pinned versions, the scripts reproduce the committed JSON, CSV and PNG files exactly. The PDF figures differ only in their embedded creation date.
