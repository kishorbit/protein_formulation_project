import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

protein_df   = pd.read_csv("data/processed/protein_features.csv")
excipient_df = pd.read_csv("data/processed/excipient_db.csv")

print(f"\nLoaded {len(protein_df)} proteins, {len(excipient_df)} excipients")

proteins    = protein_df["protein_id"].tolist()
buffers     = excipient_df[excipient_df["class"]=="buffer"]["name"].tolist()
sugars      = excipient_df[excipient_df["class"]=="sugar"]["name"].tolist()
surfactants = excipient_df[excipient_df["class"]=="surfactant"]["name"].tolist()
amino_acids = excipient_df[excipient_df["class"]=="amino_acid"]["name"].tolist()

def sample_concentration(name):
    row = excipient_df[excipient_df["name"]==name]
    if row.empty: return 0.0
    lo = float(row["conc_min_mM"].values[0])
    hi = float(row["conc_max_mM"].values[0])
    return round(random.choice([lo, (lo+hi)/2, hi]), 4)

def simulate_stability(protein_id, protein_df,
                       buffer, sugar, surfactant,
                       amino_acid, ph, temperature,
                       buf_conc, sug_conc, sur_conc, aa_conc):
    prow = protein_df[protein_df["protein_id"]==protein_id].iloc[0]

    pi          = prow["isoelectric_point"]
    instability = prow["instability_index"]
    gravy       = prow["gravy_score"]
    pct_asn     = prow["pct_asn"]
    pct_met     = prow["pct_met"]
    pct_cys     = prow["pct_cys"]
    hotspots    = prow.get("agg_hotspots", 0)
    if pd.isna(hotspots): hotspots = 0

    # ── Aggregation score (0=best, 1=worst) ──────────────────
    agg = 0.3
    agg += 0.30 * max(0, 1 - abs(ph - pi))
    agg += 0.10 * (instability / 100)
    agg += 0.10 * max(0, gravy)
    agg += 0.05 * min(hotspots / 50, 1.0)
    agg -= 0.15 if sugar in ["sucrose","trehalose"] else 0
    agg -= 0.10 if surfactant in ["polysorbate 80","polysorbate 20"] else 0
    agg -= 0.08 if amino_acid == "arginine" else 0
    agg -= 0.05 * (sug_conc / 300)
    agg += 0.10 if temperature >= 40 else 0
    agg = float(np.clip(agg + np.random.normal(0, 0.03), 0, 1))

    # ── Oxidation level ───────────────────────────────────────
    ox = 0.1
    ox += 0.30 * (pct_met / 5)
    ox += 0.20 * (pct_cys / 5)
    ox -= 0.15 if amino_acid == "methionine" else 0
    ox -= 0.10 if buffer == "histidine" else 0
    ox += 0.10 if temperature >= 40 else 0
    ox = float(np.clip(ox + np.random.normal(0, 0.02), 0, 1))

    # ── Deamidation level ─────────────────────────────────────
    deam = 0.05
    deam += 0.40 * (pct_asn / 6)
    deam += 0.10 if ph >= 7.0 else 0
    deam -= 0.10 if buffer in ["histidine","citric acid"] else 0
    deam += 0.10 if temperature >= 40 else 0
    deam = float(np.clip(deam + np.random.normal(0, 0.02), 0, 1))

    # ── Potency retention (0=worst, 1=best) ───────────────────
    potency = float(np.clip(
        1.0 - agg*0.4 - ox*0.3 - deam*0.2
        - (0.05 if temperature >= 40 else 0)
        + np.random.normal(0, 0.02), 0, 1))

    # ── Viscosity class ───────────────────────────────────────
    visc = 0
    if sug_conc > 200: visc += 1
    if aa_conc  > 100: visc += 1
    visc = int(np.clip(visc, 0, 2))

    # ── Shelf life score (0=worst, 1=best) ────────────────────
    shelf = float(np.clip(
        1.0 - (agg*0.4 + ox*0.3 + deam*0.3)
        + np.random.normal(0, 0.02), 0, 1))

    # ── NO hardcoded binary label here ───────────────────────
    # stable will be derived from composite_stability_score
    # in feature_engineering.py using configurable weights

    return {
        "aggregation_score": round(agg,   4),
        "oxidation_level":   round(ox,    4),
        "deamidation_level": round(deam,  4),
        "potency_retention": round(potency, 4),
        "viscosity_class":   visc,
        "shelf_life_score":  round(shelf, 4),
        # stable intentionally omitted — derived downstream
    }


def generate_dataset(n_per_protein=60):
    ph_values    = [5.0, 5.5, 6.0, 6.5, 7.0, 7.4]
    temperatures = [4, 25, 40]
    rows = []

    print(f"\nGenerating {len(proteins)*n_per_protein} formulation experiments...\n")

    for idx, protein_id in enumerate(proteins):
        print(f"  [{idx+1}/{len(proteins)}] {protein_id}")
        for _ in range(n_per_protein):
            buf = random.choice(buffers)
            sug = random.choice(sugars)
            sur = random.choice(surfactants)
            aa  = random.choice(amino_acids + [None])
            ph  = random.choice(ph_values)
            tmp = random.choice(temperatures)

            buf_conc = sample_concentration(buf)
            sug_conc = sample_concentration(sug)
            sur_conc = sample_concentration(sur)
            aa_conc  = sample_concentration(aa) if aa else 0.0

            outcomes = simulate_stability(
                protein_id, protein_df,
                buf, sug, sur, aa, ph, tmp,
                buf_conc, sug_conc, sur_conc, aa_conc)

            rows.append({
                "protein_id":    protein_id,
                "buffer":        buf,
                "sugar":         sug,
                "surfactant":    sur,
                "amino_acid":    aa if aa else "none",
                "ph":            ph,
                "temperature_c": tmp,
                "buf_conc_mM":   buf_conc,
                "sug_conc_mM":   sug_conc,
                "sur_conc_mM":   sur_conc,
                "aa_conc_mM":    aa_conc,
                **outcomes,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset(n_per_protein=60)
    df.to_csv("data/processed/stability_outcomes.csv", index=False)

    print(f"\n{'='*55}")
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nNote: 'stable' column removed — derived in feature_engineering.py")
    print(f"\nScore distributions:")
    for col in ["aggregation_score","oxidation_level",
                "deamidation_level","potency_retention","shelf_life_score"]:
        print(f"  {col}: mean={df[col].mean():.3f}  std={df[col].std():.3f}  "
              f"min={df[col].min():.3f}  max={df[col].max():.3f}")
    print(f"\nSaved: data/processed/stability_outcomes.csv")
