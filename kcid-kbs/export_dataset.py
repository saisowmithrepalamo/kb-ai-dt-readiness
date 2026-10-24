"""Write the seed-23 synthetic population used in the paper to a CSV file.

Uses the generator, rule layer and stratified train/test split of analysis.py
unchanged, so the file holds exactly the initiatives behind the reported results
(5,000 rows; 1,251 held out with 448 breaches). Columns are described in README.md.

Usage: python export_dataset.py   (writes kcid_synthetic_population_seed23.csv)
"""

from __future__ import annotations

import numpy as np

import analysis as A

OUT = A.ROOT / "kcid_synthetic_population_seed23.csv"
RENAME = {
    "inn": "innovation",
    "ent": "entrepreneurship",
    "data_vol": "data_volume",
    "data_sens": "data_sensitivity",
    "dt": "dt_breadth",
    "access": "access_control",
    "crypto": "encryption",
    "monitor": "monitoring",
    "respond": "incident_response",
    "skills": "security_skills",
    "vendor": "vendor_control",
    "p_true": "true_breach_probability",
    "impact": "breach_impact_I",
    "value": "opportunity_value_V",
}


def main() -> None:
    df = A.generate(np.random.default_rng(A.SEED))
    hard, soft = A.knowledge_rules(df)
    df["hard_rule"] = hard.astype(int)
    df["soft_rules_fired"] = soft

    # Same stratified split as analysis.main().
    idx = np.arange(A.N)
    rng_split = np.random.default_rng(A.SEED)
    rng_split.shuffle(idx)
    breach_idx = idx[df.loc[idx, "breach"].to_numpy() == 1]
    safe_idx = idx[df.loc[idx, "breach"].to_numpy() == 0]
    n_tr_b = int(A.TRAIN_FRAC * len(breach_idx))
    n_tr_s = int(A.TRAIN_FRAC * len(safe_idx))
    test_idx = np.concatenate([breach_idx[n_tr_b:], safe_idx[n_tr_s:]])
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    df.insert(0, "initiative_id", np.arange(1, A.N + 1))
    df = df.rename(columns=RENAME)
    test = df[df["split"] == "test"]
    assert len(test) == 1251 and int(test["breach"].sum()) == 448, "split does not match the paper"
    df.to_csv(OUT, index=False, float_format="%.6f")
    print(f"Wrote {OUT.name}: {len(df)} rows, {len(test)} held out ({int(test['breach'].sum())} breaches)")


if __name__ == "__main__":
    main()
