# Compliance checklist

Re-audit after Phase 2. The manuscript edited in this pass is `paper.md`. The submission file is `paper.docx`. The pre-edit text is `manuscript_backup_original.md`.

## Sources

| Source | Status |
|---|---|
| Official guidelines | Supplied PDF, 20 pages. Page numbers are PDF pages. |
| Journal template | Not supplied. ❓ for any template-only rule. |
| Accepted paper | Not designated in the approved request. Not used. |

A local page count used Times New Roman 12 pt, double spacing, 1-inch margins, US Letter, and figures at 6.5 inches wide. It is not Elsevier’s production engine.

## Checklist

| Requirement | Source | Status | Required fix |
|---|---|---|---|
| Short communication of at most 10 double-spaced pages, including tables and figures | PDF p. 3 | ✅ | Local render of `paper.md` is 10 pages. References begin on page 9. Do not add a paragraph. |
| Aims: a knowledge-based or other AI system for human prediction and decision-making | PDF p. 2 | ⚠️ | The system is an intelligent decision-support score plus rules. The learner is standard logistic regression on synthetic initiatives. The claim is now limited to that demonstration. A desk editor can still judge it too thin. |
| Editable `.doc`, `.docx`, or `.tex` source. PDF is not the source | PDF p. 9 | ✅ | `paper.docx` is single-column Word. `paper.md` is the working copy. |
| Word in a single column | PDF p. 9 | ✅ | `paper.docx` section size is US Letter with 1-inch margins and no second column. |
| Title page: names, full affiliation with country, corresponding author, email | PDF p. 10 | ❌ | Placeholders remain: `[Author given name(s)]`, `[Author family name]`, affiliation, and email. |
| Title avoids non-established abbreviations | PDF p. 10 | ⚠️ | The title uses “AI”. The guide allows established abbreviations. |
| Abstract ≤ 250 words; purpose, procedure, findings, conclusions; no references; abbreviation defined | PDF p. 10 | ✅ | 237 words. No citations. KCID-AI is defined. The abstract now says the three hard rules do not enter the breach process. |
| 1 to 7 keywords, English, no “and”/“of” phrases | PDF p. 10 | ✅ | Seven keywords. “Data” was removed. |
| Highlights: separate file, 3–5 bullets, ≤ 85 characters each | PDF p. 10 | ✅ | `highlights.txt`. Lengths are 67, 72, 60, and 70 characters. Encouraged, and now present. |
| Graphical abstract | Not in this PDF | ❓ | Not required by the supplied guide. Not added. |
| Equations editable, numbered in citation order, variables distinguishable | PDF p. 11 | ✅ | Equations (1)–(3) are text, cited in order, and variable names are marked for italics in `paper.md`. |
| Tables editable, numbered in order of appearance, captioned, notes below, not vertical-rule images | PDF p. 11 | ✅ | Table 1 is introduced before Table 2. Both are text in `paper.md` and Word tables in `paper.docx`. The Word style “Table Grid” draws rules. Remove borders in Word if the production editor objects. |
| Tables do not merely repeat the prose | PDF p. 11 | ⚠️ | The contrasts and intervals are in the prose because they are not columns of Table 2. The indicator correlations are in Table 1 and are no longer listed one by one in the text. |
| Figures cited in order, separate files, names such as `Figure_1` | PDF p. 11 | ✅ | `Figure_1.png`, `Figure_1.pdf`, `Figure_2.png`, `Figure_2.pdf`. Figure 1 is cited before Figure 2. |
| Caption has a title not printed on the image, plus a description | PDF p. 11 | ✅ | Both images have no title drawn on them. Captions were shortened. |
| Vector artwork as EPS or PDF | PDF p. 12 | ✅ | `Figure_1.pdf` and `Figure_2.pdf`, fonts embedded as type 42 by `analysis.py`. |
| Raster at least 300 dpi; single column ≥ 1,063 px; full page ≥ 2,244 px | PDF p. 12 | ⚠️ | Both PNGs are 300 dpi and 2,123 × 1,321 px. Submit the PDFs. The PNGs miss the full-page raster width. |
| Color usable without color vision | PDF p. 12 | ✅ | Black, vermillion, blue, and green, also separated by marker and line style. |
| Data plots come from code, not from hand-edited images | PDF p. 12 | ✅ | `analysis.py` writes both figures. Results file unchanged by the rename. |
| Option C: deposit and cite the data, or explain why sharing is impossible | PDF p. 13 | ❌ | The statement says the file regenerates the data and that deposit had not been made. Sharing is possible, so the exception does not apply. Deposit the code and outputs, then cite the repository. |
| Data statement present | PDF p. 13 | ⚠️ | A statement is present. It does not yet meet Option C. |
| Software used in the study is cited as software, and the describing article is cited as well | PDF p. 17 | ✅ | scikit-learn 1.9.1 is [10]. The 2011 article is [9]. |
| Numbered sections; abstract unnumbered | PDF p. 14 | ✅ | Sections 1–5 and 2.1–2.3. |
| Theory and calculation as separate headed sections | PDF pp. 14–15 | ❓ | The PDF describes those sections. It does not say a short communication without those headings is rejected. The material is in Sections 2 and 3. |
| Acknowledgements only in their own section before the references, if any | PDF p. 15 | ✅ | None. |
| CRediT statement | PDF p. 15 | ⚠️ | Roles are listed. The name is still a placeholder. |
| Competing-interests declarations tool, uploaded as Word | PDF pp. 5–6 | ❌ | The manuscript has a no-conflict sentence. The Elsevier declarations-tool file does not exist. The author has to complete that tool. |
| Unfunded sentence matches the guide | PDF p. 6 | ✅ | The funding section uses the guide’s sentence. |
| AI statement before the references, with the guide’s heading, naming the tool and the task | PDF pp. 6–7 | ✅ | Heading matches. The tool is named as Cursor’s AI writing assistant, with the tasks. |
| American or British English, not mixed | PDF p. 19 | ⚠️ | No full spelling pass. Visible spelling is American. |
| References numbered in order of first appearance; each in-text number has one entry and the reverse | PDF p. 16 | ✅ | First-appearance order is 1 through 12. Twelve entries. No orphans. |
| DOI where one exists; real sources | PDF p. 16 | ⚠️ | See `05_reference_report.md`. Three records were opened this pass. The others are [UNVERIFIED] here. |
| One corresponding author with contact details | PDF pp. 4 and 18 | ❌ | No name and no email. |

## Items that still block “Ready”

Corresponding author, data deposit, and the Elsevier declarations-tool file. The page count is exactly 10 in the local render.
