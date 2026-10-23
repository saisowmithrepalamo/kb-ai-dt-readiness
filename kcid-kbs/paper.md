# A knowledge-based AI decision-support system for cybersecurity-aware digital transformation using innovation and entrepreneurship data

**Manuscript type:** Short communication

**Target journal:** *Knowledge-Based Systems*

[Author given name(s)] [Author family name]

[Full affiliation, including country]

Corresponding author. Email: [corresponding author email]

## Abstract

Digital transformation initiatives are often screened on innovation, entrepreneurship, and data indicators. Those indicators do not record whether cybersecurity controls can carry the new data. This short communication specifies KCID-AI, a knowledge-based artificial intelligence decision-support system that turns that screening into one of three actions: launch, harden, or defer. A logistic score estimates breach probability from innovation, entrepreneurship, data, digital-transformation, and cybersecurity-knowledge indicators. A production-rule layer then applies hard exclusions and soft penalties for control failures. No public file records these inputs with a common breach outcome, so the system is demonstrated on a reproducible synthetic population of 5,000 initiatives whose breach label is a latent control-gap process. The three hard rules do not enter that process. On 1,251 held-out initiatives and 448 breaches, KCID-AI launches 170 breaches, compared with 198 for a logistic model trained on innovation and data indicators alone. Mean decision utility rises by 1.21 (95% bootstrap interval 0.25 to 2.19). In the pre-specified data-rich, low-access cell, launched breaches fall from 43 of 156 to 13 of 156. Adding the same cybersecurity indicators to the logistic score accounts for the utility gain. The rules then cut the remaining missed breaches in that cell from 21 of 156 to 13 of 156, with no detectable change in mean utility. Gradient boosting on the same features does not replace the knowledge layer. The population is synthetic, and the paper treats the result as a decision-system demonstration.

**Keywords:** Knowledge-based systems; Artificial intelligence; Cybersecurity; Digital transformation; Innovation; Entrepreneurship; Intelligent decision support

## 1. Introduction

Firms adopt platforms, analytics, and new digital offerings in order to change products and processes [1,2,3]. Many of those initiatives concentrate data. A screening score can then treat high innovation, entrepreneurship, and data intensity as evidence that the initiative is ready to launch, while access control, monitoring, response, and skills stay weak [4,5]. Cybersecurity frameworks describe the missing piece as outcomes an organization can state [6]. They do not supply the numeric cutoffs used below. Those cutoffs are specified for this demonstration.

KCID-AI is a decision-support system for that screening choice [7]. It does not propose a new learning algorithm. The contribution is the separation of two instruments that are often collapsed. A logistic score uses the indicators a firm can measure, including cybersecurity knowledge. A production-rule layer can force deferral or subtract a penalty when a stated control test fails. The held-out comparison asks which instrument changes utility, and which changes the remaining unsafe launches. Gradient boosting is included only as a check on flexibility.

The population is synthetic. Section 3 states the data-generating process before the results. The claim is that the knowledge layer changes the launch inside this population. It is not an estimate for a named industry.

## 2. KCID-AI

### 2.1 Screening objective

Each initiative carries eleven indicators on the unit interval. Five describe the opportunity: innovation intensity, entrepreneurship orientation, data volume, data sensitivity, and digital-transformation breadth. Six describe cybersecurity knowledge: access control, encryption, monitoring, incident response, skills, and vendor control.

*Impact* and *Value* are fixed before any model is fit:

*Impact* = 25 + 90 *Sensitivity* + 40 *Volume* + 20 *Breadth* (1)

*Value* = 18 + 35 *Innovation* + 22 *Entrepreneurship* + 12 *Breadth* (2)

Given a breach probability *p*, the screening score is

*J* = *Value* − *p* × *Impact* − *Penalty* (3)

*Penalty* is zero unless the rule layer is on. The action is launch if *J* ≥ 10, harden if −5 ≤ *J* < 10, and defer if *J* < −5. The evaluation payoff is not a cash flow. Launch receives *Value* and pays *Impact* if a breach occurs. Harden keeps 55% of *Value*, pays 5, and pays 35% of *Impact* if a breach would have occurred. Defer pays and earns nothing, so a safer action can score worse when it blocks a valuable initiative.

### 2.2 Knowledge-rule layer

The rule base is propositional and is evaluated on each initiative. Three rules are hard exclusions. If any one fires, the action is defer.

- r1: *Sensitivity* > 0.70 and access control < 0.45.
- r2: *Sensitivity* > 0.82 and encryption < 0.40.
- r3: *Volume* > 0.72 and monitoring < 0.38 and *Sensitivity* > 0.50.

Four rules are soft. Each one that fires adds 8 to *Penalty*.

- r4: *Volume* > 0.65 and monitoring < 0.42.
- r5: *Breadth* > 0.60 and incident response < 0.40.
- r6: *Entrepreneurship* > 0.70 and skills < 0.45 and *Sensitivity* > 0.50.
- r7: *Sensitivity* > 0.60 and vendor control < 0.40.

The thresholds were set as operating rules for this study. They were not searched on the test sample. Hard rules fire on 9.3% of the 5,000 initiatives. Rules r1–r3 and r4, r5, and r7 do not appear in the breach equation in Section 3. Rule r6 uses the same three cutoffs as the venture-gap term in that equation, so r6 is not an independent test of those cutoffs.

### 2.3 Probability model

The probability *p* in Eq. (3) is an L2-penalized logistic regression with *C* = 1, fit on the training sample after standardization [8]. The data feature set contains the five opportunity indicators. The knowledge set adds the six cybersecurity indicators. KCID-AI is the knowledge-set logistic probability with the rule layer on. Both this logistic model and the boosting models below are fit in scikit-learn 1.9.1 [9,10].

Two gradient-boosting benchmarks use 250 trees, a learning rate of 0.05, a maximum depth of 3, and a subsample fraction of 0.85 [11,12]. One sees the data set. The other sees the knowledge set. Neither sees the production rules. Boosting can represent interactions among the indicators it is given. It cannot apply a deferral that was never encoded as a rule.

The 2 × 2 design crosses feature set with rules. The boosting models sit outside that grid. Every model uses the action thresholds in Eq. (3).

## 3. Illustrative population

No public file reports these inputs with a breach outcome, so the benchmark uses 5,000 synthetic initiatives (seed 23). Four independent normals generate innovation, entrepreneurship, data intensity, and control maturity. Breadth also depends on innovation and data. The breach probability rises with sensitivity, volume, breadth, and entrepreneurship, falls slightly with access and monitoring, and rises on three control gaps, plus the venture-gap term that rule r6 repeats. Noise remains. The intercept sets the mean latent probability to 0.36. The realized breach rate is 0.358.

The stratified split is 3,749 / 1,251, with 448 test breaches. Fits use the training sample only. Paired bootstrap intervals use 1,000 resamples (seed 24). The cell is fixed from the training sample: data volume at least 0.493 and access control at most 0.435 (256 test initiatives, 156 breaches). Figure 1 asks whether breach risk falls with access control while a data-only score stays flat. The generator is `analysis.py`.

## 4. Results

Figure 1 uses the 656 data-rich initiatives in the test set, in access-control quintiles of 131 or 132. The observed breach rate falls from 0.718 in the lowest quintile (94 of 131; 95% Wilson interval 0.635 to 0.788) to 0.258 in the highest (34 of 132; 0.191 to 0.338). The data-only score stays between 0.434 and 0.458. The knowledge score falls from 0.651 to 0.257. Inside this slice, access control correlates with the breach label at −0.319 and with the knowledge score at −0.734. Its correlation with the data-only score is −0.049.

![Figure 1. Breach risk falls with access control. A data-only score does not.](Figure_1.png)

**Figure 1.** Breach probability by access-control quintile for 656 data-rich test initiatives. Points are observed rates with 95% Wilson intervals. Dashed: data-only logistic mean probability. Solid: knowledge-set logistic mean probability. Quintile sizes are 131, 131, 131, 131, and 132.

Table 1 shows why innovation and entrepreneurship do not substitute for that control. Their correlations with breach are about zero. Data volume and data sensitivity are the opportunity indicators that move with breach. Every cybersecurity indicator moves the other way.

**Table 1. What the screening indicators measure (N = 5,000)**

| Indicator | Block | Mean (SD) | Correlation with breach |
|---|---|---:|---:|
| Innovation intensity | Innovation and entrepreneurship data | 0.503 (0.189) | −0.002 |
| Entrepreneurship orientation | Innovation and entrepreneurship data | 0.494 (0.182) | 0.019 |
| Data volume | Innovation and entrepreneurship data | 0.497 (0.193) | 0.208 |
| Data sensitivity | Innovation and entrepreneurship data | 0.498 (0.202) | 0.282 |
| Digital-transformation breadth | Innovation and entrepreneurship data | 0.502 (0.205) | 0.073 |
| Access control | Cybersecurity knowledge | 0.499 (0.207) | −0.240 |
| Encryption | Cybersecurity knowledge | 0.497 (0.183) | −0.228 |
| Monitoring | Cybersecurity knowledge | 0.498 (0.194) | −0.214 |
| Incident response | Cybersecurity knowledge | 0.499 (0.177) | −0.224 |
| Skills | Cybersecurity knowledge | 0.501 (0.152) | −0.180 |
| Vendor control | Cybersecurity knowledge | 0.499 (0.161) | −0.209 |

*Note.* Correlations are population values from the synthetic design in Section 3.

On the standardized knowledge-set logistic, a one-standard-deviation higher data sensitivity multiplies the odds of breach by 1.72, data volume by 1.31, access control by 0.65, and encryption by 0.82. Once access and encryption are in the score, the partial coefficients on monitoring, response, skills, and vendor control are small. Table 1 still shows those four indicators correlated with breach. Their remaining weight is in the threshold rules.

Table 2 reports the actions. The launch rate barely changes: 60.9% for the data-only logistic and 60.6% for KCID-AI. Of 448 breaches, those models launch 198 and 170. The missed-breach difference is −0.062 (95% interval −0.102 to −0.024). Mean utility moves from 20.05 to 21.26, a difference of 1.21 (0.25 to 2.19). In the cell, launched breaches fall from 43 of 156 to 13 of 156 (−0.192; −0.257 to −0.131). Of the launches inside the cell, 43 of 98 data-only launches are breaches, against 13 of 43 for KCID-AI.

**Table 2. Held-out screening decisions (n = 1,251; 448 breaches)**

| Model | Features | Rules | AUC | Launched breaches | Breach rate among launches | Mean utility | Launched breaches in the cell |
|---|---|---|---:|---:|---:|---:|---:|
| Data-only logistic | Opportunity | Off | 0.691 | 198/448 | 0.260 | 20.05 | 43/156 |
| Data-only boosting | Opportunity | Off | 0.677 | 227/448 | 0.275 | 19.65 | 52/156 |
| Knowledge logistic | Opportunity + cybersecurity | Off | 0.728 | 180/448 | 0.232 | 21.27 | 21/156 |
| Knowledge boosting | Opportunity + cybersecurity | Off | 0.714 | 211/448 | 0.251 | 20.83 | 23/156 |
| Data logistic + rules | Opportunity | On | 0.691 | 180/448 | 0.247 | 20.27 | 33/156 |
| KCID-AI | Opportunity + cybersecurity | On | 0.728 | 170/448 | 0.224 | 21.26 | 13/156 |

*Note.* The cell contains 256 test initiatives with data volume at or above 0.493 and access control at or below 0.435, of which 156 are breaches. Area under the curve is identical within a feature set because the rules change the action, not the probability. Paired intervals in the text use 1,000 resamples of the test set.

Adding the cybersecurity indicators, with rules still off, raises utility by 1.22 (0.30 to 2.14) and cuts the cell’s launched breaches from 43 to 21. The missed-breach rate falls by 0.040 (−0.078 to −0.004). Turning the rules on after those features are in the score leaves mean utility unchanged (−0.006; −0.28 to 0.25) and cuts launched breaches from 180 to 170 (−0.022; −0.037 to −0.011) and from 21 to 13 in the cell (−0.050; −0.089 to −0.020). Rules on the data-only score also reach 180 launched breaches, at a utility of 20.27. The features account for the utility gain. The rules remove a further set of launches.

Gradient boosting does not close that gap. On the opportunity indicators it launches 227 of 448 breaches, at utility 19.65. On the knowledge set it launches 211. Its utility difference against KCID-AI includes zero (0.43; −0.29 to 1.23). Its cell difference does not (−0.064; −0.108 to −0.021).

Figure 2 restricts Figure 1 to initiatives that do breach. In the lowest access quintile, 94 breaches, the data-only logistic launches 0.287, the knowledge logistic 0.096, and KCID-AI 0.043. In the highest quintile, 34 breaches, those shares are 0.382, 0.676, and 0.676. Launches move toward the quintiles where Figure 1 says the risk has fallen. The rules bind on the left of Figure 2, where KCID-AI separates from the knowledge logistic, and they are silent on the right, where the two knowledge curves meet.

![Figure 2. Launches shift from low-access breaches toward high-access breaches.](Figure_2.png)

**Figure 2.** Share of breaches launched in the same quintiles as Figure 1. Counts are 94, 65, 50, 41, and 34. KCID-AI adds the rule layer to the knowledge logistic.

## 5. Conclusion

Cybersecurity indicators raise held-out utility. The rules do not, once those indicators are already in the score. They do cut further launched breaches in the data-rich, low-access cell, from 43 of 156 to 13 of 156, and they state why the initiative is deferred. Those counts describe this simulation, not a sector. Rule r6 repeats a term in the breach process. The other rules do not. A penalty other than 8, or cutoffs other than 10 and −5, was not varied. The learner is standard logistic regression. The claim is the readable score, the readable rules, and the finding that boosting does not replace either.

## Data availability

The population, the split, both figures, and both tables are produced by `analysis.py` with the seeds stated in Section 3. The synthetic records can be regenerated from that file. They are not deposited in a public repository. Deposit is possible, and it had not been made at the time of this manuscript.

## Declaration of competing interest

The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this manuscript.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author used Cursor’s AI writing assistant to draft language, organize the analysis code, and format the manuscript. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article. The tables and both figures are outputs of `analysis.py`.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## CRediT authorship contribution statement

[Author family name]: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review and editing, Visualization.

## References

[1] G. Vial, Understanding digital transformation: a review and a research agenda, Journal of Strategic Information Systems 28 (2019) 118–144. https://doi.org/10.1016/j.jsis.2019.01.003

[2] S. Nambisan, Digital entrepreneurship: toward a digital technology perspective of entrepreneurship, Entrepreneurship Theory and Practice 41 (2017) 1029–1055. https://doi.org/10.1111/etap.12254

[3] A. Bharadwaj, O.A. El Sawy, P.A. Pavlou, N. Venkatraman, Digital business strategy: toward a next generation of insights, MIS Quarterly 37 (2013) 471–482. https://doi.org/10.25300/MISQ/2013/37:2.3

[4] L.A. Gordon, M.P. Loeb, The economics of information security investment, ACM Transactions on Information and System Security 5 (2002) 438–457. https://doi.org/10.1145/581271.581274

[5] R. von Solms, J. van Niekerk, From information security to cyber security, Computers & Security 38 (2013) 97–102. https://doi.org/10.1016/j.cose.2013.04.004

[6] C. Pascoe, S. Quinn, K. Scarfone, The NIST Cybersecurity Framework (CSF) 2.0, NIST Cybersecurity White Paper 29, National Institute of Standards and Technology, Gaithersburg, MD, 2024. https://doi.org/10.6028/NIST.CSWP.29

[7] S.H. Liao, Expert system methodologies and applications — a decade review from 1995 to 2004, Expert Systems with Applications 28 (2005) 93–103. https://doi.org/10.1016/j.eswa.2004.08.003

[8] T. Hastie, R. Tibshirani, J. Friedman, The Elements of Statistical Learning, second ed., Springer, New York, 2009.

[9] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, É. Duchesnay, Scikit-learn: machine learning in Python, Journal of Machine Learning Research 12 (2011) 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

[10] F. Pedregosa, et al., scikit-learn version 1.9.1 [software], n.d. https://scikit-learn.org/ (accessed 23 October 2026).

[11] J.H. Friedman, Greedy function approximation: a gradient boosting machine, Annals of Statistics 29 (2001) 1189–1232. https://doi.org/10.1214/aos/1013203451

[12] A.L. Buczak, E. Guven, A survey of data mining and machine learning methods for cyber security intrusion detection, IEEE Communications Surveys & Tutorials 18 (2016) 1153–1176. https://doi.org/10.1109/COMST.2015.2494502
