import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

COMPOSITE_WEIGHTS = {
    "aggregation_score":  0.35,
    "oxidation_level":    0.25,
    "deamidation_level":  0.20,
    "potency_retention":  0.15,
    "shelf_life_score":   0.05,
}
STABILITY_THRESHOLD = 0.40

TEXT_COLS = {"protein_id","protein_name","gene","query_label","sequence"}

print("\nLoading data files...")
protein_df   = pd.read_csv("data/processed/protein_features.csv")
excipient_df = pd.read_csv("data/processed/excipient_db.csv")
stability_df = pd.read_csv("data/processed/stability_outcomes.csv")

print(f"  Protein features:    {protein_df.shape}")
print(f"  Excipient database:  {excipient_df.shape}")
print(f"  Stability outcomes:  {stability_df.shape}")

if "stable" in stability_df.columns:
    print("  Dropping hardcoded stable column — will be re-derived")
    stability_df = stability_df.drop(columns=["stable"])

# ── Composite score + derived label ──────────────────────
print("\nComputing composite stability score...")
stability_df["composite_stability_score"] = (
    COMPOSITE_WEIGHTS["aggregation_score"]  * stability_df["aggregation_score"]
  + COMPOSITE_WEIGHTS["oxidation_level"]    * stability_df["oxidation_level"]
  + COMPOSITE_WEIGHTS["deamidation_level"]  * stability_df["deamidation_level"]
  + COMPOSITE_WEIGHTS["potency_retention"]  * (1 - stability_df["potency_retention"])
  + COMPOSITE_WEIGHTS["shelf_life_score"]   * (1 - stability_df["shelf_life_score"])
).round(4)

stability_df["stable"] = (
    stability_df["composite_stability_score"] < STABILITY_THRESHOLD
).astype(int)

stable_rate = stability_df["stable"].mean() * 100
print(f"  Stable rate: {stable_rate:.1f}%")
print(f"  Composite — mean: {stability_df['composite_stability_score'].mean():.3f}  "
      f"std: {stability_df['composite_stability_score'].std():.3f}")

# ── Merge protein features ────────────────────────────────
protein_merge_cols = [c for c in protein_df.columns
                      if c not in TEXT_COLS or c == "protein_id"]
df = stability_df.merge(
    protein_df[protein_merge_cols], on="protein_id", how="left")

# ── Merge excipient properties ────────────────────────────
exc_props = excipient_df[[
    "name","mol_weight","logp","hbd","hba","tpsa","rotatable_bonds"
]].copy()

for role in ["buffer","sugar","surfactant","amino_acid"]:
    renamed = exc_props.rename(columns={
        "name":            role,
        "mol_weight":      f"{role}_mol_weight",
        "logp":            f"{role}_logp",
        "hbd":             f"{role}_hbd",
        "hba":             f"{role}_hba",
        "tpsa":            f"{role}_tpsa",
        "rotatable_bonds": f"{role}_rotbonds",
    })
    df = df.merge(renamed, on=role, how="left")

# ── Encode categoricals (LabelEncoder only — no one-hot) ──
cat_cols = ["buffer","sugar","surfactant","amino_acid","protein_id"]
le = LabelEncoder()
for col in cat_cols:
    df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))

# ── Interaction features ──────────────────────────────────
df["sug_x_instability"] = df["sug_conc_mM"] * df["instability_index"]
df["sur_x_gravy"]       = df["sur_conc_mM"] * df["gravy_score"].abs()
df["buf_x_asn"]         = df["buf_conc_mM"] * df["pct_asn"]
df["sug_x_cys"]         = df["sug_conc_mM"] * df["pct_cys"]
df["high_temp_flag"]    = (df["temperature_c"] >= 40).astype(int)
df["agg_x_temp"]        = df["agg_mean"] * df["high_temp_flag"]
df["agg_x_gravy"]       = df["agg_mean"] * df["gravy_score"]
df["hotspot_x_sug"]     = df["agg_hotspot_frac"] * df["sug_conc_mM"]

def get_dist_from_pi(row):
    col = f"dist_from_pI_at_pH_{row['ph']}"
    return row[col] if col in row.index else np.nan

df["dist_from_pI_formulation"] = df.apply(get_dist_from_pi, axis=1)
df["ph_risk_flag"] = (df["dist_from_pI_formulation"] < 0.5).astype(int)

# ── Fill known missing patterns ───────────────────────────
surfactant_fill = {
    "surfactant_mol_weight": 1310.0, "surfactant_logp":    4.5,
    "surfactant_hbd":           0.0, "surfactant_hba":    26.0,
    "surfactant_tpsa":         69.0, "surfactant_rotbonds":80.0,
}
for col, val in surfactant_fill.items():
    df[col] = df[col].fillna(val)

# ── Save — NO imputation, NO scaling here ────────────────
# Imputation happens inside each LOO fold in held_out_validation.py
# and ml_model.py to prevent any leakage from test proteins.
df.to_csv("data/processed/dataset_merged.csv", index=False)

print(f"\n{'='*55}")
print(f"Merged dataset shape: {df.shape}")
print(f"Missing values remaining (will be handled per fold):")

all_targets = [
    "aggregation_score","oxidation_level","deamidation_level",
    "potency_retention","shelf_life_score",
    "composite_stability_score","stable","viscosity_class",
]
text_and_cat = TEXT_COLS | set(cat_cols) | {"ph","temperature_c"}
feature_cols = [
    c for c in df.columns
    if c not in all_targets
    and c not in text_and_cat
    and df[c].dtype in [np.float64, np.float32, np.int64,
                        np.int32, np.int16, np.uint8, bool]
]
missing = df[feature_cols].isnull().sum()
missing = missing[missing > 0]
if len(missing):
    print(missing.to_string())
else:
    print("  None — clean dataset")

print(f"\nFeature columns identified: {len(feature_cols)}")
print(f"NOTE: No imputation applied here — leakage-free")
print(f"Saved: data/processed/dataset_merged.csv")

# Save feature column list for downstream scripts
pd.Series(feature_cols).to_csv(
    "data/processed/feature_cols.csv", index=False, header=False)
print(f"Saved: data/processed/feature_cols.csv")
