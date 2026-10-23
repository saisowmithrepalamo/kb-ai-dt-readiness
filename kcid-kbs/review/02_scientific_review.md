# Scientific review

Phase 1 only. No experimental number was changed.

Manuscript: `kcid-kbs/paper.md`. Code: `kcid-kbs/analysis.py`. Numbers rechecked in this pass against `results.json`, `table1_descriptives.csv`, and `table2_performance.csv`. Figure pixel sizes were read from the PNG headers.

## 1. Gap and novelty

The decision problem inside the simulation is clear. A score that sees innovation, entrepreneurship, and data can miss a breach gradient that moves with access control. Figure 1 is the right way to show that problem. The conclusion is honest that the learner is ordinary logistic regression and that the population is synthetic.

The novelty claim is not yet supported at the level the aims on PDF p. 2 ask for: original, innovative, and creative research, and new development of a knowledge-based system.

- The introduction cites digital-transformation and entrepreneurship sources for the setting, Gordon and Loeb (2002) for the economics of security investment, and Liao (2005) for expert-system surveys. None of those is a current knowledge-based cybersecurity screening system. The gap against recent knowledge-based and intrusion-decision work is not established. Buczak and Guven (2016) is cited where boosting is introduced, not as the literature the system extends.
- NIST CSF 2.0 is cited for the sentence that operators already state thresholds of the kind used in rules r1–r7. The cutoffs, including sensitivity > 0.70 and access < 0.45, are constants in `knowledge_rules` (`analysis.py`, the hard-rule and soft-rule block). NIST does not state those cutoffs. The citation over-claims the source of the rules.
- Bharadwaj, El Sawy, Pavlou, and Venkatraman (2013) is an editorial on digital business strategy. It is grouped with Vial (2019) and Nambisan (2017) as support for firms adopting platforms and new offerings. That is a loose use of the source. Its DOI, `10.25300/MISQ/2013/37:2.3`, matches the publisher record. The other nine reference records were not re-resolved in this pass and are [UNVERIFIED] until the reference report.
- The title and the keyword list name cybersecurity, digital transformation, innovation, and entrepreneurship. The experiment uses those names as synthetic inputs. It does not study a transformation program, a venture, or an observed breach.
- The conclusion already limits the claim to the decision system. The introduction’s threefold contribution should stay at that size.

Verdict: the gap inside this simulation is explicit. The gap against the journal’s literature is not. This is a system demonstration. It is not yet an original method contribution.

## 2. Methodology against the code

These claims match `analysis.py` and `results.json`.

| Claim in the manuscript | Where it is in the code or output |
|---|---|
| Impact = 25 + 90 Sensitivity + 40 Volume + 20 Breadth | `generate`, the `impact` line |
| Value = 18 + 35 Innovation + 22 Entrepreneurship + 12 Breadth | `generate`, the `value` line |
| J = Value − p × Impact − Penalty; launch if J ≥ 10; harden if J ≥ −5; otherwise defer | `decide` |
| Soft penalty of 8 per fired rule; a hard rule forces defer | `decide` |
| Rules r1–r7 and their cutoffs | `knowledge_rules` |
| L2 logistic, C = 1, standardized, training sample only | `fit_proba` |
| Boosting: 250 trees, learning rate 0.05, depth 3, subsample 0.85 | `GradientBoostingClassifier` in `fit_proba` |
| N = 5,000, seed 23, mean latent probability calibrated to 0.36 | `generate`. `mean_p_true` is 0.360. `breach_rate` is 0.3578, printed as 0.358 |
| Stratified 3,749 / 1,251 and 448 test breaches | `n_train`, `n_test`. 198 / 0.4419642857142857 = 448 |
| Training median of data volume 0.493 and training 40th percentile of access 0.435 | `vol_cut` 0.492824 and `access_cut` 0.434578 |
| Cell of 256 initiatives and 156 breaches | `table2_performance.csv` |
| Hard rules on 9.3% of the population | `hard_rate` 0.0934 |
| Bootstrap of 1,000 paired resamples | `N_BOOT = 1000` |
| Harden keeps 55% of value, costs 5, and pays 35% of impact if a breach occurs | `utility` |

Mismatches and omissions:

1. **Shared cutoffs between the label and rule r6.** Section 3 says the breach probability is not the rule base. That is true of the three hard rules. It is not fully true of the venture term. `venture_gap` uses entrepreneurship > 0.70, skills < 0.45, and sensitivity > 0.50, and the latent equation adds `1.35 * venture_gap`. Soft rule r6 uses the same three cutoffs. The manuscript discloses the latent term and does not disclose that r6 repeats it. The rule is partly aligned with the label by construction.

2. **“Cross-factor dependence is weak” is underspecified.** The four latent normals are drawn independently. Digital-transformation breadth is `0.40 * innovation + 0.30 * data` plus noise. Dependence is weak across the four factors and is not weak for breadth. The next sentence does say breadth loads on innovation and data. No numerical mismatch.

3. **“Before the test metrics are read.”** In the current script the cell cuts are computed from the training sample before the test metrics are summarized. There is no separate pre-registration file. A reader cannot verify the order of discovery except by trusting the script. This is a validity limit, not a disagreement with the current file.

4. **Odds ratios are not an output field.** They were recomputed here from `logistic_coefs`: exp(0.543529) = 1.722, exp(0.268985) = 1.309, exp(−0.428163) = 0.652, exp(−0.194497) = 0.823. The printed 1.72, 1.31, 0.65, and 0.82 match. The script does not write the odds ratios into `results.json`.

5. **Bootstrap seed.** Resamples use `SEED + 1` (24). The manuscript says seed 23 for the population and is silent on the bootstrap seed. Not a false number.

No other equation, hyperparameter, split size, or threshold in the manuscript disagrees with the code.

## 3. Do the reported numbers trace to the outputs?

Yes, except one rounded interval end.

Table 1 means, standard deviations, and correlations match `table1_descriptives.csv` at the printed precision. Innovation’s correlation −0.002101 prints as −0.002. Skills’ standard deviation 0.152087 prints as 0.152. Access control’s correlation −0.239927 prints as −0.240.

Table 2 matches `table2_performance.csv`:

| Model | Launched breaches | Utility | Cell |
|---|---|---|---|
| Data-only logistic | 198/448 from rate 0.441964 | 20.0517 → 20.05 | 43/156 from 0.275641 |
| Data-only boosting | 227/448 from 0.506696 | 19.6479 → 19.65 | 52/156 |
| Knowledge logistic | 180/448 from 0.401786 | 21.2668 → 21.27 | 21/156 |
| Knowledge boosting | 211/448 from 0.470982 | 20.8345 → 20.83 | 23/156 |
| Data logistic + rules | 180/448 | 20.2750 → 20.27 | 33/156 |
| KCID-AI | 170/448 from 0.379464 | 21.2628 → 21.26 | 13/156 |

AUC values 0.691, 0.677, 0.728, and 0.714 match at three decimals. Launch shares 60.9% and 60.6% match 0.609113 and 0.605915. The cell launch counts, 43 of 98 and 13 of 43, match `cell_launches` 98 and 43.

Figure 1 matches `figure_risk` and `separation`. There are 656 data-rich test initiatives. Quintile sizes are 131, 131, 131, 131, and 132. Observed rates are 0.718 and 0.258. Wilson intervals are 0.635–0.788 and 0.191–0.338. Data-only means span 0.434 to 0.458. Knowledge means run from 0.651 to 0.257. Correlations are −0.319, −0.734, and −0.049. Both PNGs are 2,123 × 1,321 pixels at 300 dpi.

Figure 2 matches `figure_launch`: 0.287, 0.096, and 0.043 in Q1; 0.382 and 0.676 in Q5; breach counts 94, 65, 50, 41, and 34.

Contrasts match `results.json` except one bound:

| Contrast | Manuscript | `results.json` | Status |
|---|---|---|---|
| KCID − data-only, missed-breach rate | −0.062 (−0.102 to −0.024) | −0.06186 (−0.10207 to −0.02396) | Match at 3 decimals |
| KCID − data-only, utility | 1.21 (0.25 to 2.19) | 1.2137 (0.2536 to 2.1941) | Match at 2 decimals |
| KCID − data-only, cell | −0.192 (−0.257 to −0.131) | −0.1915 (−0.25714 to −0.13124) | Match |
| Knowledge logistic − data-only, utility | 1.22 (0.30 to 2.14) | 1.2193 (0.3026 to 2.1365) | Match |
| Knowledge logistic − data-only, missed-breach rate | −0.040 (−0.078 to −0.004) | −0.03963 (−0.07814 to −0.00441) | Match |
| KCID − knowledge logistic, missed-breach rate | −0.022 (−0.037 to −0.011) | −0.02223 (−0.03680 to −0.01075) | Match |
| KCID − knowledge logistic, utility | −0.006 (−0.28 to 0.25) | −0.00565 (−0.28186 to 0.25480) | Match |
| KCID − knowledge logistic, cell | −0.050 (−0.089 to −0.021) | −0.05033 (−0.08862 to −0.02026) | Mismatch. −0.02026 rounds to −0.020, not −0.021 |
| KCID − knowledge boosting, utility | 0.43 (−0.29 to 1.23) | 0.4286 (−0.2901 to 1.2270) | Match |
| KCID − knowledge boosting, cell | −0.064 (−0.108 to −0.021) | −0.06372 (−0.10828 to −0.02068) | Match. −0.02068 rounds to −0.021 |

Do not change any other digit. The only proven error is that one interval end.

## 4. Discussion, baselines, and validity

What the design already does well:

- Every model uses the same action thresholds, so the comparison is not an artifact of a different cutoff.
- The 2 × 2 separates cybersecurity features from the symbolic rules. Adding the features raises utility by 1.22, and that interval excludes zero. Adding the rules after the features changes utility by −0.006, and that interval includes zero. The prose states that split correctly.
- Boosting is a real flexibility check, and it does not remove the cell errors the rules remove.
- The conclusion states the synthetic population, the specified thresholds, and the ordinary learner.

What is missing or over-strong:

- **Constructed safety.** Because r6’s cutoffs also enter the label, the rule layer is not an independent test of those three conditions. The hard rules do not enter the label and are the cleaner test. The results do not separate those two kinds of rule.
- **No observed breach data.** A synthetic recovery shows that the score can see a signal planted in access control. It does not show that the signal exists in firms.
- **No security baseline beyond logistic regression and gradient boosting.** There is no non-learned control checklist, no cost-sensitive classifier, and no other calibrated probability model.
- **Thresholds are untested.** The conclusion says a different penalty than 8, or cutoffs other than 10 and −5, would move actions. That sensitivity is not in `results.json`. Leaving it as a limitation is acceptable only if the claim stays descriptive.
- **Utility is an author-built payoff.** The 0.55 retention, the cost of 5, and the 0.35 residual impact are not estimated. Mean utility can be moved by those constants. The manuscript says they are not cash flows. The results still treat the utility difference as a main finding. That finding is conditional on the payoff.
- **The last sentence of the conclusion is ahead of the evidence.** The 43-of-156 versus 13-of-156 counts are real for this file. They are not evidence about firms.
- **“Defined before the test metrics are read”** should be softened unless a record outside the analysis script shows the cell was fixed first.

## 5. Short communication eligibility

| Limit | Result |
|---|---|
| Type | A compact system note can be a short communication. The present length cannot. |
| 10 double-spaced pages, including figures and tables (PDF p. 3) | Not met on the word-count estimate. Exact pages need a double-spaced render. |
| Figure and table count | The PDF states no separate cap of two. Two figures and two tables are inside the page cap, not a cap of their own. |
| Reference cap | No numerical cap in the PDF. Ten references is not a violation. Their order is. |
| Fit to the aims (PDF p. 2) | Partial. Intelligent decision support is in scope. A synthetic logistic score with hand-written penalties is thin for “original, innovative and creative” research unless the claim stays a methods demonstration and the length is cut. |

## 6. Phase 1 judgment

The experiment file and the manuscript agree, apart from the cell-interval end −0.021, which should be −0.020 at three decimals. The scientific problem is not arithmetic. It is that the paper asks this journal to treat a fully synthetic, partly self-aligned rule test as a cybersecurity decision system for digital transformation, innovation, and entrepreneurship.

Formatting blockers that are already certain from the PDF: the page estimate is over the short-communication cap, there are eight keywords, the references are out of citation order, there is no named author or corresponding email, the manuscript is Markdown rather than Word or LaTeX, the funding sentence does not match the guide, the AI statement does not name the tool, the software is not cited, and the research data are not deposited.

No manuscript edit was made. Approval is required before Phase 2.
