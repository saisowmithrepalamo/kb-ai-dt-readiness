# Figures and tables

Checked against `results.json`, `table1_descriptives.csv`, and `table2_performance.csv` after Phase 2. The rerun that renamed the figures left `results.json` byte-for-byte identical.

## Figure 1

| Item | Value |
|---|---|
| Files | `Figure_1.png`, `Figure_1.pdf` |
| Script | `analysis.py`, function `_save_figures` |
| Source fields | `figure_risk`, `separation` |
| PNG | 2,123 × 1,321 pixels, 300 dpi |
| Status | ✅ |

Values in the caption and in Section 4: 656 data-rich test initiatives. Quintile sizes 131, 131, 131, 131, 132. Observed rates 0.718 (94 of 131; Wilson 0.635 to 0.788) and 0.258 (34 of 132; 0.191 to 0.338). Data-only means span 0.434 to 0.458. Knowledge means run from 0.651 to 0.257. Correlations −0.319, −0.734, and −0.049. All match the JSON at the printed precision.

## Figure 2

| Item | Value |
|---|---|
| Files | `Figure_2.png`, `Figure_2.pdf` |
| Script | `analysis.py`, function `_save_figures` |
| Source field | `figure_launch` |
| PNG | 2,123 × 1,321 pixels, 300 dpi |
| Status | ✅ |

Q1 (94 breaches): data-only 0.287, knowledge logistic 0.096, KCID-AI 0.043. Q5 (34 breaches): 0.382, 0.676, and 0.676. Breach counts 94, 65, 50, 41, 34. All match `figure_launch`.

## Table 1

| Item | Value |
|---|---|
| Location | `paper.md` and `paper.docx` |
| Source | `table1_descriptives.csv` |
| Status | ✅ |

Means, standard deviations, and correlations match the CSV at three decimals. Examples: innovation correlation −0.002101 prints as −0.002. Access correlation −0.239927 prints as −0.240. Skills standard deviation 0.152087 prints as 0.152.

## Table 2

| Item | Value |
|---|---|
| Location | `paper.md` and `paper.docx` |
| Source | `table2_performance.csv` and `results.json` `metrics` |
| Status | ✅ |

| Model | Launched breaches | Utility | Cell |
|---|---|---|---|
| Data-only logistic | 198/448 from 0.441964 | 20.05 | 43/156 |
| Data-only boosting | 227/448 from 0.506696 | 19.65 | 52/156 |
| Knowledge logistic | 180/448 from 0.401786 | 21.27 | 21/156 |
| Knowledge boosting | 211/448 from 0.470982 | 20.83 | 23/156 |
| Data logistic + rules | 180/448 | 20.27 | 33/156 |
| KCID-AI | 170/448 from 0.379464 | 21.26 | 13/156 |

AUC values 0.691, 0.677, 0.728, and 0.714 match at three decimals.

Contrasts in the prose:

| Contrast | Printed | JSON | Status |
|---|---|---|---|
| KCID − data-only, missed-breach rate | −0.062 (−0.102 to −0.024) | −0.06186 (−0.10207 to −0.02396) | ✅ |
| KCID − data-only, utility | 1.21 (0.25 to 2.19) | 1.2137 (0.2536 to 2.1941) | ✅ |
| KCID − data-only, cell | −0.192 (−0.257 to −0.131) | −0.1915 (−0.25714 to −0.13124) | ✅ |
| Knowledge − data-only, utility | 1.22 (0.30 to 2.14) | 1.2193 (0.3026 to 2.1365) | ✅ |
| Knowledge − data-only, missed-breach rate | −0.040 (−0.078 to −0.004) | −0.03963 (−0.07814 to −0.00441) | ✅ |
| KCID − knowledge, utility | −0.006 (−0.28 to 0.25) | −0.00565 (−0.28186 to 0.25480) | ✅ |
| KCID − knowledge, missed-breach rate | −0.022 (−0.037 to −0.011) | −0.02223 (−0.03680 to −0.01075) | ✅ |
| KCID − knowledge, cell | −0.050 (−0.089 to −0.020) | −0.05033 (−0.08862 to −0.02026) | ✅ corrected |
| KCID − knowledge boosting, utility | 0.43 (−0.29 to 1.23) | 0.4286 (−0.2901 to 1.2270) | ✅ |
| KCID − knowledge boosting, cell | −0.064 (−0.108 to −0.021) | −0.06372 (−0.10828 to −0.02068) | ✅ |

Launch shares 60.9% and 60.6% match 0.609113 and 0.605915. Cell launch quality, 43 of 98 and 13 of 43, matches `cell_launches` 98 and 43.
