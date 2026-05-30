import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────
# 1. Load outputs from Step 1 and Step 2
# ─────────────────────────────────────────
protein_df   = pd.read_csv("data/processed/protein_features.csv")
excipient_df = pd.read_csv("data/processed/excipient_db.csv")

proteins   = protein_df["protein_id"].tolist()
excipients = excipient_df["name"].tolist()

buffers     = excipient_df[excipient_df["class"] == "buffer"]["name"].tolist()
sugars      = excipient_df[excipient_df["class"] == "sugar"]["name"].tolist()
surfactants = excipient_df[excipient_df["class"] == "surfactant"]["name"].tolist()
amino_acids = excipient_df[excipient_df["class"] == "amino_acid"]["name"].tolist()


def sample_concentration(name, excipient_df):
    row = excipient_df[excipient_df["name"] == name]
    if row.empty:
        return 0.0
    lo = float(row["conc_min_mM"].values[0])
    hi = float(row["conc_max_mM"].values[0])
    return round(random.choice([lo, (lo + hi) / 2, hi]), 4)


# ─────────────────────────────────────────
# 2. Stability outcome simulator
#    FIX: 'stable' label is NO LONGER hardcoded here.
#    It will be DERIVED from the composite score after
#    regression — keeping it physically consistent.
# ─────────────────────────────────────────
def simulate_stability(protein_id, protein_df,
                       buffer, sugar, surfactant,
                       amino_acid, ph, temperature,
                       buf_conc, sug_conc, sur_conc, aa_conc):

    prow = protein_df[protein_df["protein_id"] == protein_id].iloc[0]

    pi          = prow["isoelectric_point"]
    instability = prow["instability_index"]
    gravy       = prow["gravy_score"]
    pct_asn     = prow["pct_asn"]
    pct_met     = prow["pct_met"]
    pct_cys     = prow["pct_cys"]

    # --- Aggregation score (0=none, 1=severe) ---
    agg = 0.3
    agg += 0.3 * max(0, 1 - abs(ph - pi))
    agg += 0.1 * (instability / 100)
    agg += 0.1 * max(0, gravy)
    agg -= 0.15 if sugar in ["sucrose", "trehalose"] else 0
    agg -= 0.10 if surfactant in ["polysorbate 80", "polysorbate 20"] else 0
    agg -= 0.08 if amino_acid == "arginine" else 0
    agg -= 0.05 * (sug_conc / 300)
    agg += 0.10 if temperature >= 40 else 0
    agg = float(np.clip(agg + np.random.normal(0, 0.03), 0, 1))

    # --- Oxidation level (0=none, 1=severe) ---
    ox = 0.1
    ox += 0.3 * (pct_met / 5)
    ox += 0.2 * (pct_cys / 5)
    ox -= 0.15 if amino_acid == "methionine" else 0
    ox -= 0.10 if "EDTA" in [buffer, amino_acid] else 0
    ox += 0.10 if temperature >= 40 else 0
    ox = float(np.clip(ox + np.random.normal(0, 0.02), 0, 1))

    # --- Deamidation level (0=none, 1=severe) ---
    deam = 0.05
    deam += 0.4 * (pct_asn / 6)
    deam += 0.1 if ph >= 7.0 else 0
    deam -= 0.10 if buffer in ["histidine", "citric acid"] else 0
    deam += 0.10 if temperature >= 40 else 0
    deam = float(np.clip(deam + np.random.normal(0, 0.02), 0, 1))

    # --- Potency retention (0=lost, 1=full) ---
    potency = 1.0
    potency -= agg * 0.4
    potency -= ox * 0.3
    potency -= deam * 0.2
    potency -= 0.05 if temperature >= 40 else 0
    potency = float(np.clip(potency + np.random.normal(0, 0.02), 0, 1))

    # --- Viscosity class ---
    visc = 0
    if sug_conc > 200: visc += 1
    if aa_conc > 100:  visc += 1
    visc = int(np.clip(visc, 0, 2))

    # --- Shelf life score (0=poor, 1=excellent) ---
    shelf = 1.0 - (agg * 0.4 + ox * 0.3 + deam * 0.3)
    shelf = float(np.clip(shelf + np.random.normal(0, 0.02), 0, 1))

    # ─────────────────────────────────────────────────────────
    # FIX: Composite score is now the PRIMARY output.
    # Weights are tunable — defaults reflect standard pharma
    # priorities (aggregation most critical, shelf life least).
    # 'stable' is DERIVED by thresholding the composite score,
    # NOT by hardcoded individual thresholds.
    # This ensures physical consistency always.
    # ─────────────────────────────────────────────────────────
    WEIGHTS = {
        "agg":   0.35,   # aggregation: highest risk
        "ox":    0.20,   # oxidation
        "deam":  0.20,   # deamidation
        "pot":   0.15,   # potency retention
        "shelf": 0.10,   # shelf life
    }
    STABILITY_THRESHOLD = 0.55   # tunable by scientist

    composite = (
        (1 - agg)   * WEIGHTS["agg"]  +
        (1 - ox)    * WEIGHTS["ox"]   +
        (1 - deam)  * WEIGHTS["deam"] +
        potency     * WEIGHTS["pot"]  +
        shelf       * WEIGHTS["shelf"]
    )
    composite = float(np.clip(composite, 0, 1))

    # Binary label DERIVED from composite — not hardcoded
    stable = int(composite >= STABILITY_THRESHOLD)

    return {
        "aggregation_score": round(agg,      4),
        "oxidation_level":   round(ox,       4),
        "deamidation_level": round(deam,     4),
        "potency_retention": round(potency,  4),
        "viscosity_class":   visc,
        "shelf_life_score":  round(shelf,    4),
        "composite_score":   round(composite, 4),   # NEW: stored for reference
        "stable":            stable,                 # DERIVED not hardcoded
    }


# ─────────────────────────────────────────
# 3. Generate dataset
# ─────────────────────────────────────────
def generate_dataset(n_per_protein=80):
    rows = []
    ph_values    = [5.0, 5.5, 6.0, 6.5, 7.0, 7.4]
    temperatures = [4, 25, 40]

    for protein_id in proteins:
        for _ in range(n_per_protein):
            buf  = random.choice(buffers)
            sug  = random.choice(sugars)
            sur  = random.choice(surfactants)
            aa   = random.choice(amino_acids + [None])
            ph   = random.choice(ph_values)
            temp = random.choice(temperatures)

            buf_conc = sample_concentration(buf, excipient_df)
            sug_conc = sample_concentration(sug, excipient_df)
            sur_conc = sample_concentration(sur, excipient_df)
            aa_conc  = sample_concentration(aa, excipient_df) if aa else 0.0

            outcomes = simulate_stability(
                protein_id, protein_df,
                buf, sug, sur, aa, ph, temp,
                buf_conc, sug_conc, sur_conc, aa_conc
            )

            row = {
                "protein_id":    protein_id,
                "buffer":        buf,
                "sugar":         sug,
                "surfactant":    sur,
                "amino_acid":    aa if aa else "none",
                "ph":            ph,
                "temperature_c": temp,
                "buf_conc_mM":   buf_conc,
                "sug_conc_mM":   sug_conc,
                "sur_conc_mM":   sur_conc,
                "aa_conc_mM":    aa_conc,
                **outcomes
            }
            rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 4. Run and save
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\nGenerating stability dataset...\n")
    df = generate_dataset(n_per_protein=80)
    df.to_csv("data/processed/stability_outcomes.csv", index=False)

    print(f"✓ Dataset shape: {df.shape}")
    print(f"\nStable vs unstable (derived from composite score):")
    print(df["stable"].value_counts().to_string())
    print(f"\nComposite score distribution:")
    print(df["composite_score"].describe().round(3).to_string())
    print(f"\nMean outcomes per target:")
    print(df[[
        "aggregation_score", "oxidation_level",
        "deamidation_level", "potency_retention",
        "shelf_life_score",  "composite_score"
    ]].mean().round(3).to_string())
    print(f"\nSaved to: data/processed/stability_outcomes.csv")
