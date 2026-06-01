import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os

print("\nLoading data...")
merged = pd.read_csv("data/processed/dataset_merged.csv")
prot   = pd.read_csv("data/processed/protein_features.csv")

print(f"  merged: {merged.shape}  prot: {prot.shape}")

# ── Join protein features if not already in merged ────────
prot_cols_needed = ["protein_id","pct_cys","pct_met","pct_asn",
                    "instability_index","sequence_length",
                    "isoelectric_point","gravy_score"]
prot_cols_needed = [c for c in prot_cols_needed if c in prot.columns]
prot_sub = prot[prot_cols_needed].drop_duplicates("protein_id")

# only merge columns not already present
new_prot_cols = [c for c in prot_cols_needed
                 if c != "protein_id" and c not in merged.columns]
if new_prot_cols:
    merged = merged.merge(prot_sub[["protein_id"] + new_prot_cols],
                          on="protein_id", how="left")
    print(f"  Joined {new_prot_cols} from protein_features.csv")
else:
    print("  All protein feature columns already present in merged")

# ── New interaction features ──────────────────────────────
print("\nEngineering interaction features...")

new_feats = {}

# --- Oxidation mechanism features
# Met oxidation is strongly temperature + pH driven
if "pct_met" in merged.columns and "temperature_c" in merged.columns:
    new_feats["met_x_temp"]    = merged["pct_met"] * merged["temperature_c"]
if "pct_met" in merged.columns and "ph" in merged.columns:
    new_feats["met_x_ph"]      = merged["pct_met"] * merged["ph"]
# Cys oxidation driven by pH (disulfide formation rate) and temperature
if "pct_cys" in merged.columns and "ph" in merged.columns:
    new_feats["cys_x_ph"]      = merged["pct_cys"] * merged["ph"]
if "pct_cys" in merged.columns and "temperature_c" in merged.columns:
    new_feats["cys_x_temp"]    = merged["pct_cys"] * merged["temperature_c"]
# Cys/Met ratio — distinguishes oxidation mechanism
if "pct_cys" in merged.columns and "pct_met" in merged.columns:
    denom = merged["pct_met"].replace(0, np.nan)
    new_feats["cys_met_ratio"] = (merged["pct_cys"] / denom).fillna(0).clip(0, 50)
    new_feats["cys_met_sum"]   = merged["pct_cys"] + merged["pct_met"]

# --- Deamidation mechanism features
# Asn deamidation accelerated by high pH and temperature
if "pct_asn" in merged.columns and "ph" in merged.columns:
    new_feats["asn_x_ph"]      = merged["pct_asn"] * merged["ph"]
if "pct_asn" in merged.columns and "temperature_c" in merged.columns:
    new_feats["asn_x_temp"]    = merged["pct_asn"] * merged["temperature_c"]
# Instability × Asn — unstable proteins expose Asn residues more
if "instability_index" in merged.columns and "pct_asn" in merged.columns:
    new_feats["instab_x_asn"]  = merged["instability_index"] * merged["pct_asn"]

# --- Shelf life / potency stability features
# Long unstable sequences degrade faster
if "instability_index" in merged.columns and "sequence_length" in merged.columns:
    new_feats["instab_x_seqlen"] = (merged["instability_index"]
                                    * np.log1p(merged["sequence_length"]))
# Charge state at formulation pH (already have dist_from_pI_formulation?)
if "dist_from_pI_formulation" not in merged.columns:
    if "isoelectric_point" in merged.columns and "ph" in merged.columns:
        new_feats["dist_pI_x_ph"] = abs(merged["isoelectric_point"]
                                         - merged["ph"]) * merged["ph"]

# --- High-risk binary flags (from diagnosis)
if "pct_cys" in merged.columns:
    cys_thresh = prot["pct_cys"].mean() + 2 * prot["pct_cys"].std()
    new_feats["flag_high_cys"]  = (merged["pct_cys"] > cys_thresh).astype(int)
    print(f"  flag_high_cys  threshold={cys_thresh:.2f}  "
          f"n={new_feats['flag_high_cys'].sum()}")

if "pct_met" in merged.columns:
    met_thresh = prot["pct_met"].mean() + 2 * prot["pct_met"].std()
    new_feats["flag_high_met"]  = (merged["pct_met"] > met_thresh).astype(int)
    print(f"  flag_high_met  threshold={met_thresh:.2f}  "
          f"n={new_feats['flag_high_met'].sum()}")

if "pct_asn" in merged.columns:
    asn_thresh = prot["pct_asn"].mean() + 2 * prot["pct_asn"].std()
    new_feats["flag_high_asn"]  = (merged["pct_asn"] > asn_thresh).astype(int)
    print(f"  flag_high_asn  threshold={asn_thresh:.2f}  "
          f"n={new_feats['flag_high_asn'].sum()}")

if "instability_index" in merged.columns:
    new_feats["flag_unstable"]  = (merged["instability_index"] > 60).astype(int)
    print(f"  flag_unstable  threshold=60.00  "
          f"n={new_feats['flag_unstable'].sum()}")

if "sequence_length" in merged.columns:
    seq_thresh = prot["sequence_length"].mean() + 2 * prot["sequence_length"].std()
    new_feats["flag_long_seq"]  = (merged["sequence_length"] > seq_thresh).astype(int)
    print(f"  flag_long_seq  threshold={seq_thresh:.0f}  "
          f"n={new_feats['flag_long_seq'].sum()}")

# ── Add to merged and save ────────────────────────────────
print(f"\nAdding {len(new_feats)} new features:")
for col, vals in new_feats.items():
    merged[col] = vals
    print(f"  + {col:<25}  mean={merged[col].mean():.4f}  "
          f"std={merged[col].std():.4f}")

merged.to_csv("data/processed/dataset_merged.csv", index=False)
print(f"\n  Saved: data/processed/dataset_merged.csv  "
      f"shape={merged.shape}")

# ── Also update protein_features.csv with flag columns ───
flag_cols = [c for c in new_feats if c.startswith("flag_")]
if flag_cols:
    flag_df = merged[["protein_id"] + flag_cols].drop_duplicates("protein_id")
    prot_out = prot.merge(flag_df, on="protein_id", how="left")
    for c in flag_cols:
        if c in prot_out.columns:
            prot_out[c] = prot_out[c].fillna(0).astype(int)
    prot_out.to_csv("data/processed/protein_features.csv", index=False)
    print(f"  Saved: data/processed/protein_features.csv  "
          f"shape={prot_out.shape}")

# ── Correlation check: do new features correlate with targets? ──
print("\nCorrelation of new features with targets:")
TARGETS = ["aggregation_score","oxidation_level","deamidation_level",
           "potency_retention","shelf_life_score"]
present_targets = [t for t in TARGETS if t in merged.columns]

print(f"\n  {'Feature':<25} " +
      "  ".join(f"{t.replace('_level','').replace('_score','')[:9]:>9}"
                for t in present_targets))
print("  " + "-"*75)

for feat in new_feats:
    if feat.startswith("flag_"):
        continue
    corrs = []
    for t in present_targets:
        sub = merged[[feat, t]].dropna()
        c   = np.corrcoef(sub[feat], sub[t])[0,1] if len(sub) > 5 else np.nan
        corrs.append(c)
    strong = any(abs(c) > 0.15 for c in corrs if not np.isnan(c))
    marker = " <--" if strong else ""
    print(f"  {feat:<25} " +
          "  ".join(f"{c:>9.3f}" if not np.isnan(c) else f"{'nan':>9}"
                    for c in corrs) + marker)

print(f"\n{'='*60}")
print("FEATURE ENGINEERING COMPLETE")
print("="*60)
print(f"""
  {len(new_feats)} new features added to dataset_merged.csv

  Oxidation features : met_x_temp, met_x_ph, cys_x_ph,
                       cys_x_temp, cys_met_ratio, cys_met_sum
  Deamidation features: asn_x_ph, asn_x_temp, instab_x_asn
  Stability features : instab_x_seqlen, dist_pI_x_ph
  Risk flags         : flag_high_cys, flag_high_met,
                       flag_high_asn, flag_unstable, flag_long_seq

  Next step: re-run your model training script with the
  updated dataset_merged.csv to pick up the new features.
""")
print("="*60)
