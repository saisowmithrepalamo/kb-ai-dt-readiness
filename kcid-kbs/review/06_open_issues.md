# Open issues

## Needs the author

1. Replace every placeholder with the given name, family name, full affiliation including country, and the corresponding email. The CRediT line needs the same family name.
2. Deposit `analysis.py`, `results.json`, and the two figure files, then replace the data-availability paragraph with the repository citation. The current paragraph says deposit had not been made. That does not meet Option C, because sharing is possible.
3. Complete Elsevier’s declarations tool and upload its Word file. The manuscript sentence is not a substitute.
4. Confirm that “Cursor’s AI writing assistant” is the tool name you want in the published AI statement. If a different product did part of the drafting, add it. Do not leave the statement generic.
5. If you add any sentence, recheck the page count. The local double-spaced render is exactly 10 pages.

## Not verified

- DOIs and page ranges for references [1], [2], [4], [5], [7], [11], and [12]. See `05_reference_report.md`.
- A DOI for Hastie, Tibshirani, and Friedman (2009). None is printed.
- The release date of scikit-learn 1.9.1. The software reference uses n.d. and an access date.
- Elsevier’s LaTeX template. It was not supplied and was not opened.
- Whether Elsevier’s converter will also count this file as 10 pages. The local render used Times New Roman 12 pt, double spacing, 1-inch margins, and US Letter.

## Scientific limits that were disclosed, not removed

- The population is synthetic. The last paragraph says the 43-of-156 and 13-of-156 counts describe the simulation.
- Rule r6 uses the same three cutoffs as the venture-gap term in the breach equation. The paper now says so. The other rules do not enter that equation.
- Mean utility depends on the author-chosen payoff: 55% of value retained, a cost of 5, and 35% of impact after hardening. Those constants were not varied.
- The penalty of 8 and the score cutoffs 10 and −5 were not varied.
- There is no observed breach file and no security baseline beyond logistic regression, gradient boosting, and the rule layer.
- NIST CSF 2.0 does not supply the numeric cutoffs.

## Repository

`journal-revision` was not created. The manuscript folder is not its own git repository, and the only repository above it is the home directory. Branching that repository would have included unrelated personal files. The pre-edit manuscript is `manuscript_backup_original.md`.
