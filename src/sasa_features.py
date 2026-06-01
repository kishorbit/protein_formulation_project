import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("\nLoading data...")
df_prot = pd.read_csv("data/processed/protein_features.csv")
df_merged = pd.read_csv("data/processed/dataset_merged.csv")

print(f"  Protein features shape: {df_prot.shape}")
print(f"  Merged dataset shape:   {df_merged.shape}")

# ── SASA proxy features ───────────────────────────────────
# True SASA requires 3D structure (not available).
# Proxy: sliding-window hydrophilicity exposure score.
# Logic: Met/Trp surrounded by hydrophilic residues
# are more solvent-exposed → higher oxidation risk.
# Window of ±3 residues around each Met/Trp.

HYDROPHILICITY = {
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,
    'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
    'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,
    'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,
}

def sasa_proxy(sequence, target_aa, window=3):
    """
    For each target residue (Met or Trp), compute mean
    hydrophilicity of surrounding window. Lower surrounding
    hydrophilicity = more buried = lower oxidation risk.
    Returns: mean exposure score across all target residues.
    """
    if not isinstance(sequence, str) or len(sequence) == 0:
        return np.nan
    seq = sequence.upper()
    scores = []
    for i, aa in enumerate(seq):
        if aa == target_aa:
            start = max(0, i - window)
            end   = min(len(seq), i + window + 1)
            neighbors = [seq[j] for j in range(start, end) if j != i]
            if neighbors:
                neighbor_score = np.mean([
                    HYDROPHILICITY.get(n, 0) for n in neighbors
                ])
                scores.append(neighbor_score)
    return np.mean(scores) if scores else 0.0

def burial_fraction(sequence, target_aa, window=3):
    """
    Fraction of target residues that appear buried
    (surrounded by hydrophobic neighbors, mean > 1.5).
    """
    if not isinstance(sequence, str) or len(sequence) == 0:
        return np.nan
    seq = sequence.upper()
    buried = 0
    total  = 0
    for i, aa in enumerate(seq):
        if aa == target_aa:
            total += 1
            start = max(0, i - window)
            end   = min(len(seq), i + window + 1)
            neighbors = [seq[j] for j in range(start, end) if j != i]
            if neighbors:
                score = np.mean([HYDROPHILICITY.get(n,0) for n in neighbors])
                if score > 1.5:
                    buried += 1
    return buried / total if total > 0 else 0.0

def terminal_exposure(sequence, target_aa, terminal_window=15):
    """
    Fraction of target residues within 15 aa of N/C terminus.
    Terminal residues are typically more solvent-exposed.
    """
    if not isinstance(sequence, str) or len(sequence) == 0:
        return np.nan
    seq = sequence.upper()
    n = len(seq)
    exposed = sum(1 for i,aa in enumerate(seq)
                  if aa==target_aa and
                  (i < terminal_window or i >= n - terminal_window))
    total = seq.count(target_aa)
    return exposed / total if total > 0 else 0.0

print("\nComputing SASA proxy features...")
seq_col = None
for c in ["sequence","protein_sequence","seq"]:
    if c in df_prot.columns:
        seq_col = c
        break

if seq_col is None:
    print("  No sequence column found. Checking columns:")
    print(f"  {list(df_prot.columns)}")
    raise SystemExit(1)

print(f"  Using sequence column: '{seq_col}'")
valid_seqs = df_prot[seq_col].notna().sum()
print(f"  Proteins with sequences: {valid_seqs} / {len(df_prot)}")

df_prot["met_exposure_score"]   = df_prot[seq_col].apply(
    lambda s: sasa_proxy(s, "M"))
df_prot["trp_exposure_score"]   = df_prot[seq_col].apply(
    lambda s: sasa_proxy(s, "W"))
df_prot["met_buried_frac"]      = df_prot[seq_col].apply(
    lambda s: burial_fraction(s, "M"))
df_prot["trp_buried_frac"]      = df_prot[seq_col].apply(
    lambda s: burial_fraction(s, "W"))
df_prot["met_terminal_frac"]    = df_prot[seq_col].apply(
    lambda s: terminal_exposure(s, "M"))
df_prot["trp_terminal_frac"]    = df_prot[seq_col].apply(
    lambda s: terminal_exposure(s, "W"))
df_prot["ox_risk_composite"]    = (
    df_prot["met_exposure_score"].fillna(0) * -1 +
    df_prot["trp_exposure_score"].fillna(0) * -1 +
    df_prot["met_terminal_frac"].fillna(0) +
    df_prot["trp_terminal_frac"].fillna(0)
)

new_features = [
    "met_exposure_score","trp_exposure_score",
    "met_buried_frac","trp_buried_frac",
    "met_terminal_frac","trp_terminal_frac",
    "ox_risk_composite",
]

print(f"\n  New SASA features computed:")
for f in new_features:
    col = df_prot[f]
    print(f"    {f:<25} mean={col.mean():.3f}  "
          f"std={col.std():.3f}  null={col.isna().sum()}")

# ── Merge new features into dataset_merged ────────────────
print("\nMerging SASA features into dataset...")
merge_cols = ["protein_id"] + new_features
df_sasa = df_prot[merge_cols].drop_duplicates("protein_id")

df_updated = df_merged.merge(df_sasa, on="protein_id", how="left")

print(f"  Rows before: {len(df_merged)}")
print(f"  Rows after:  {len(df_updated)}")
print(f"  New columns: {len(df_updated.columns) - len(df_merged.columns)}")

# ── Update feature_cols.csv ───────────────────────────────
old_feature_cols = pd.read_csv(
    "data/processed/feature_cols.csv", header=None)[0].tolist()
old_feature_cols = [c for c in old_feature_cols if c in df_updated.columns]

added = [f for f in new_features if f in df_updated.columns
         and f not in old_feature_cols]
new_feature_cols = old_feature_cols + added

print(f"\n  Feature cols before: {len(old_feature_cols)}")
print(f"  Features added:      {len(added)}")
print(f"  Feature cols after:  {len(new_feature_cols)}")

# ── Save ──────────────────────────────────────────────────
df_prot.to_csv("data/processed/protein_features.csv", index=False)
df_updated.to_csv("data/processed/dataset_merged.csv", index=False)
pd.Series(new_feature_cols).to_csv(
    "data/processed/feature_cols.csv", header=False, index=False)

print(f"\n{'='*55}")
print("SASA FEATURES COMPLETE")
print("="*55)
print(f"  Features added: {added}")
print(f"  dataset_merged.csv updated")
print(f"  feature_cols.csv updated")
print(f"  protein_features.csv updated")
print("="*55)
