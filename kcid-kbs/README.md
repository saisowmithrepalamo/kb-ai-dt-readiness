# KCID-AI: code and data

Code, simulated data and result files for the article **Breach-risk models and security rules for screening digital transformation initiatives** (Sai Sowmith Repala), prepared for *Knowledge-Based Systems*.

All data here are simulated. No real organisation, person or breach record is included. The manuscript itself is not posted here while it is under consideration at the journal.

## Files

| File | What it is |
|---|---|
| [`kcid_synthetic_population_seed23.csv`](kcid_synthetic_population_seed23.csv) | The simulated dataset behind every seed-23 result: 5,000 initiatives, 1,251 held out (448 breaches). Columns are described below |
| `analysis.py`, `sensitivity.py`, `extensions.py` | The study: data generator, rule layer, models, policies and evaluation |
| `export_dataset.py` | Writes the dataset above, using `analysis.py` unchanged |
| `results.json`, `sensitivity_results.json`, `extensions_results.json` | Every number reported in the paper |
| `table1_descriptives.csv`, `table2_performance.csv`, `payoff_sensitivity.csv`, `generator_variants.csv`, `joint_sensitivity.csv` | Result tables written by the scripts |
| `Figure_1`–`Figure_4` (`.png`, `.pdf`) | The scripts' own plots of the results |
| `highlights.txt` | The article's highlights |
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
| `analysis.py` | `results.json`, `table1_descriptives.csv`, `table2_performance.csv`, `Figure_1`, `Figure_2` | Table 2 (first six rows), Table 3 (first three comparators), Table A.1, Fig. 2 |
| `sensitivity.py` | `sensitivity_results.json`, `payoff_sensitivity.csv`, `Figure_3` | Table 2 (remaining six rows), Table 3 (remaining four comparators), Table 4, Fig. B.1 |
| `extensions.py` | `extensions_results.json`, `generator_variants.csv`, `joint_sensitivity.csv`, `Figure_4` | Tables 5, 6 and C.1, Fig. C.1 |
| `export_dataset.py` | `kcid_synthetic_population_seed23.csv` | The data behind all seed-23 results |

The rules of Table 1 are defined in `knowledge_rules()` in `analysis.py`. Fig. 1 of the paper is a diagram of the method, not a result. The paper redraws Figs. 2, B.1 and C.1 at journal size from the same JSON values.

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
python export_dataset.py   # a few seconds
```

With the pinned versions, the scripts reproduce the committed JSON, CSV and PNG files exactly. The PDF figures differ only in their embedded creation date.
