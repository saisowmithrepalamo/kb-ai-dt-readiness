# Change log

Backup before editing: `manuscript_backup_original.md`. No experimental output was regenerated except the figure filenames. `results.json` after that rerun is identical to the pre-edit file.

Only one reported number changed. For KCID-AI minus the knowledge logistic, the cell-rate interval upper end is −0.020. It had been printed as −0.021. `results.json` has −0.02026, which rounds to −0.020. The boosting comparison that also ends near −0.021 was left as −0.021, because −0.02068 rounds to −0.021.

| Section | Change | Reason |
|---|---|---|
| Title page | Placeholders now ask for given name, family name, full affiliation with country, and the corresponding email | PDF p. 10. The name was not invented. |
| Abstract | “Not the rules themselves” replaced by the statement that the three hard rules do not enter the breach process | The old sentence was broader than the generator. Word count is 237. |
| Keywords | “Data” removed. Seven keywords remain | PDF p. 10 caps keywords at 7. “Data” stays in the title. |
| Introduction | Citations renumbered to first-appearance order. NIST is cited for outcomes an organization can state, not for the numeric cutoffs | PDF p. 16. The cutoffs are author-specified. |
| Contribution | The introduction states that the paper does not propose a new learning algorithm | Matches the evidence. |
| Section 2.2 | Rule r6 is identified as the same three cutoffs as the venture-gap term. The other rules are identified as absent from the breach equation | The old text said the label was not the rule base. That was incomplete. |
| Section 2.3 | scikit-learn 1.9.1 and the 2011 article are cited | PDF p. 17. |
| Section 3 | “Before the test metrics are read” removed. The cell is described as fixed from the training sample. Bootstrap seed 24 is stated. The generator paragraph is shorter | The script does not prove the historical order of discovery. Seed 24 is `SEED + 1` in `analysis.py`. Length. |
| Section 3 | The forward mention of Table 2 was removed | Tables must be numbered in order of appearance. |
| Section 4 | Indicator-by-indicator correlations are left in Table 1 rather than repeated | PDF p. 11. |
| Section 4 | Cell interval for rules added after the knowledge score corrected to −0.089 to −0.020 | Output file. |
| Section 5 | The closing claim is limited to this simulation. The untested penalty and cutoffs are stated as not varied | The 43-of-156 and 13-of-156 counts are real for the file and are not firm estimates. |
| Captions | Shortened. The counts remain in the captions and in Section 4 | PDF p. 11 asks captions to stay short. Needed for the 10-page render. |
| Funding | Replaced with the guide’s sentence | PDF p. 6. |
| AI declaration | Heading set to the guide’s heading. Tool named | PDF pp. 6–7. |
| References | Bharadwaj moved to [3]. Gordon and Loeb, von Solms, and NIST follow. NIST authors set to Pascoe, Quinn, and Scarfone from the NIST record. Pedregosa article is [9]. Software is [10]. Friedman is [11]. Buczak is [12] | Citation order, and a verified NIST author list. |
| Figures | `analysis.py` now writes `Figure_1` and `Figure_2` as PNG and PDF. Old filenames removed | PDF p. 11. Images were not edited by hand. |
| Highlights | `highlights.txt` added | PDF p. 10. |
| Word file | `paper.docx` added, single column, double spaced | PDF p. 9. |
