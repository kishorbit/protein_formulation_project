import numpy as np
import pandas as pd
import requests, io, time


# ─────────────────────────────────────────────────────────
# Biophysical scales
# ─────────────────────────────────────────────────────────

KD_HYDROPHOBICITY = {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}
CF_BETA = {
    'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
    'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
    'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
    'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70,
}
CHARGE_PH7 = {
    'A':  0.0, 'R':  1.0, 'N':  0.0, 'D': -1.0, 'C':  0.0,
    'Q':  0.0, 'E': -1.0, 'G':  0.0, 'H':  0.1, 'I':  0.0,
    'L':  0.0, 'K':  1.0, 'M':  0.0, 'F':  0.0, 'P':  0.0,
    'S':  0.0, 'T':  0.0, 'W':  0.0, 'Y':  0.0, 'V':  0.0,
}
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid_sequence(s) -> bool:
    if not isinstance(s, str): return False
    s = s.strip().upper()
    if len(s) < 10: return False
    return sum(1 for c in s if c in VALID_AA) / len(s) >= 0.80

# ─────────────────────────────────────────────────────────
# Per-residue scoring
# ─────────────────────────────────────────────────────────

def per_residue_score(sequence: str, ph: float = 6.0) -> np.ndarray:
    seq  = sequence.strip().upper()
    n    = len(seq)
    hydro  = np.array([KD_HYDROPHOBICITY.get(aa, 0.0) for aa in seq])
    beta   = np.array([CF_BETA.get(aa, 1.0)           for aa in seq])
    charge = np.array([CHARGE_PH7.get(aa, 0.0)        for aa in seq])
    for i, aa in enumerate(seq):
        if aa == 'H':
            charge[i] = max(0.0, 1.0 - (ph - 5.0) / 3.0)
    def norm(a):
        r = a.max() - a.min()
        return a if r == 0 else 2*(a-a.min())/r - 1
    raw      = 0.5*norm(hydro) + 0.3*norm(beta) - 0.2*norm(charge)
    half     = 2
    padded   = np.pad(raw, half, mode='edge')
    windowed = np.array([padded[i:i+5].mean() for i in range(n)])
    return windowed * 3.0

def compute_agg_features(sequence: str, ph: float = 6.0) -> dict:
    scores = per_residue_score(sequence, ph=ph)
    n_hot  = int((scores > 1.0).sum())
    return {
        "agg_mean":         round(float(scores.mean()), 4),
        "agg_min":          round(float(scores.min()),  4),
        "agg_max":          round(float(scores.max()),  4),
        "agg_std":          round(float(scores.std()),  4),
        "agg_hotspots":     n_hot,
        "agg_hotspot_frac": round(n_hot / max(len(sequence), 1), 4),
    }

# ─────────────────────────────────────────────────────────
# Validation on known proteins
# ─────────────────────────────────────────────────────────

KNOWN = {
    "Abeta42 (high risk)":      ("DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA", "high"),
    "IgG1 Fc fragment (low)":   ("APELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPE", "low"),
    "Lysozyme (medium)":        ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQA", "medium"),
}

def validate():
    print("Validation on known proteins:")
    print(f"  {'Protein':<30} {'Expected':>8} {'agg_mean':>9} {'hotspots':>9} {'Result':>8}")
    print(f"  {'-'*68}")
    all_pass = True
    for name, (seq, exp) in KNOWN.items():
        f    = compute_agg_features(seq)
        mean = f["agg_mean"]
        hot  = f["agg_hotspots"]
        ok   = ((exp=="high" and mean>0.3) or
                (exp=="low"  and mean<0.2) or
                 exp=="medium")
        result = "PASS" if ok else "CHECK"
        if not ok: all_pass = False
        print(f"  {name:<30} {exp:>8} {mean:>9.3f} {hot:>9} {result:>8}")
    print(f"  {'Overall: ALL PASS' if all_pass else 'Overall: some checks needed'}\n")

# ─────────────────────────────────────────────────────────
# Update protein_features.csv — safe column handling
# ─────────────────────────────────────────────────────────

# Columns that must NEVER be treated as numeric
TEXT_COLS = {"protein_id", "protein_name", "gene",
             "query_label", "sequence"}

def update_protein_features():
    path = "data/processed/protein_features.csv"
    df   = pd.read_csv(path)

    print(f"Loaded protein_features.csv: {df.shape}")
    print(f"Text columns (will be preserved): "
          f"{[c for c in TEXT_COLS if c in df.columns]}")

    # Drop old fake columns
    fake = ["camsol_mean","camsol_min","camsol_hotspots"]
    df   = df.drop(columns=[c for c in fake if c in df.columns])

    # Drop existing agg columns so we don't duplicate
    existing_agg = [c for c in df.columns if c.startswith("agg_")]
    df = df.drop(columns=existing_agg)

    print(f"\nComputing real aggregation scores for {len(df)} proteins...\n")
    print(f"  {'#':<4} {'protein_id':<14} {'name':<35} "
          f"{'agg_mean':>9} {'hotspots':>9} {'frac':>7}")
    print(f"  {'-'*82}")

    agg_rows = []
    for i, row in df.iterrows():
        pid   = row["protein_id"]
        pname = str(row.get("protein_name",""))[:33]
        seq   = row.get("sequence","")

        if not is_valid_sequence(seq):
            print(f"  {i+1:<4} {pid:<14} {pname:<35} "
                  f"{'no sequence':>9}")
            agg_rows.append({k: np.nan for k in
                ["agg_mean","agg_min","agg_max",
                 "agg_std","agg_hotspots","agg_hotspot_frac"]})
            continue

        feats = compute_agg_features(seq, ph=6.0)
        agg_rows.append(feats)
        print(f"  {i+1:<4} {pid:<14} {pname:<35} "
              f"{feats['agg_mean']:>+9.3f} "
              f"{feats['agg_hotspots']:>9} "
              f"{feats['agg_hotspot_frac']:>6.1%}")

    agg_df = pd.DataFrame(agg_rows, index=df.index)
    df     = pd.concat([df, agg_df], axis=1)
    df.to_csv(path, index=False)

    print(f"\n{'='*55}")
    print(f"Summary statistics:")
    print(f"  Proteins processed:  {len(df)}")
    print(f"  Valid sequences:     {agg_df['agg_mean'].notna().sum()}")
    print(f"  Mean agg_mean:       {agg_df['agg_mean'].mean():.3f}")
    print(f"  Mean hotspot frac:   {agg_df['agg_hotspot_frac'].mean():.1%}")
    print(f"\nNew columns added:")
    for c in ["agg_mean","agg_min","agg_max",
              "agg_std","agg_hotspots","agg_hotspot_frac"]:
        print(f"  {c}")
    print(f"\nSaved: {path}")
    return df

if __name__ == "__main__":
    validate()
    df = update_protein_features()
