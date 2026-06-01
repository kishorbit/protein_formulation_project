import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

os.makedirs("data/processed", exist_ok=True)

protein_df   = pd.read_csv("data/processed/protein_features_expanded.csv")
excipient_df = pd.read_csv("data/processed/excipient_db.csv")

proteins    = protein_df["protein_id"].tolist()
buffers     = excipient_df[excipient_df["class"] == "buffer"]["name"].tolist()
sugars      = excipient_df[excipient_df["class"] == "sugar"]["name"].tolist()
surfactants = excipient_df[excipient_df["class"] == "surfactant"]["name"].tolist()
amino_acids = excipient_df[excipient_df["class"] == "amino_acid"]["name"].tolist()
salts       = excipient_df[excipient_df["class"] == "salt"]["name"].tolist()

PROTEIN_CONCS       = [1, 10, 100]
TEMPERATURES        = [4, 25, 37, 40, 50]
PH_VALUES           = [5.0, 5.5, 6.0, 6.5, 7.0, 7.4]
STABILITY_THRESHOLD = 0.68
N_NORMAL            = 110
N_STRESS            = 40

def sample_conc(name):
    if name is None:
        return 0.0
    row = excipient_df[excipient_df["name"] == name]
    if row.empty:
        return 0.0
    lo = float(row["conc_min_mM"].values[0])
    hi = float(row["conc_max_mM"].values[0])
    return round(random.uniform(lo, hi), 4)

def simulate_stability(prow, buf, sug, sur, aa, salt,
                       ph, temp, protein_conc,
                       buf_conc, sug_conc, sur_conc,
                       aa_conc, salt_conc):

    pi           = prow["isoelectric_point"]
    instability  = prow["instability_index"]       # 30-73
    gravy        = prow["gravy_score"]              # -0.81 to +0.23
    pct_asn      = prow["pct_asn"]                 # PERCENT: 0-8.5
    pct_met      = prow["pct_met"]                 # PERCENT: 0.8-6.3
    pct_cys      = prow["pct_cys"]                 # PERCENT: 0-6.5
    met_exp_frac = float(prow["met_exposed_fraction"]) if pd.notna(prow["met_exposed_fraction"]) else 0.5
    ox_mean_rsa  = float(prow["ox_mean_rsa"])          if pd.notna(prow["ox_mean_rsa"])          else 0.38
    agg_hotspot  = float(prow["agg_hotspot_frac"])     if pd.notna(prow["agg_hotspot_frac"])     else 0.05

    # ── AGGREGATION ───────────────────────────────────────
    agg = 0.10

    # Intrinsic baseline — some proteins aggregate even under mild conditions
    # High instability index + basic pI (hard to formulate at pH 5-7.4)
    agg += 0.06 * (instability / 100.0)
    if pi > 8.0:   agg += 0.08   # basic proteins: hard to stay away from pI
    elif pi < 5.5: agg += 0.04   # very acidic: limited pH window

    # pI proximity (DLVO — electrostatic repulsion lost at pI)
    pH_dist = abs(ph - pi)
    agg += 0.24 * np.exp(-pH_dist / 1.5)

    # Intrinsic instability (normalized 0-1)
    agg += 0.07 * (instability / 100.0)

    # Hydrophobicity (only positive GRAVY increases agg)
    agg += 0.07 * max(0.0, gravy / 0.25)

    # Hotspot fraction (normalized to max observed 0.13)
    agg += 0.10 * min(agg_hotspot / 0.13, 1.0)

    # Concentration — non-linear second order kinetics
    conc_factor = (protein_conc / 100.0) ** 1.4
    agg += 0.18 * conc_factor

    # Temperature — tiered
    if temp >= 50:
        agg += 0.25
    elif temp >= 40:
        agg += 0.13
    elif temp >= 37:
        agg += 0.06
    else:
        agg += 0.01 * (temp - 4) / 21.0

    # Ionic strength
    agg += 0.04 * (salt_conc / 200.0)

    # Stabilizers
    if sug in ["sucrose", "trehalose"]:
        agg -= 0.14 * min(sug_conc / 300.0, 1.0)
    elif sug in ["mannitol", "sorbitol"]:
        agg -= 0.06 * min(sug_conc / 300.0, 1.0)

    if sur == "polysorbate 80":
        agg -= 0.18 * min(sur_conc / 0.08, 1.0)   # strongest interface protection
    elif sur == "polysorbate 20":
        agg -= 0.14 * min(sur_conc / 0.08, 1.0)   # slightly weaker than PS80
    elif sur == "poloxamer 188":
        agg -= 0.10 * min(sur_conc / 0.05, 1.0)   # bulk protection, less interface

    if aa == "arginine":
        agg -= 0.12 * min(aa_conc / 150.0, 1.0)
    elif aa == "proline":
        agg -= 0.05 * min(aa_conc / 150.0, 1.0)
    elif aa == "glycine":
        agg -= 0.04 * min(aa_conc / 200.0, 1.0)

    agg = float(np.clip(agg + np.random.normal(0, 0.025), 0, 1))

    # ── OXIDATION ─────────────────────────────────────────
    # Target range: 0.05-0.20 optimal, 0.25-0.55 stressed
    ox = 0.04

    # Exposed Met is primary risk
    ox += 0.18 * met_exp_frac                    # max +0.18

    # Overall oxidizable surface
    ox += 0.08 * ox_mean_rsa                     # max +0.06

    # Sequence content (pct in PERCENT, normalize by max=6)
    ox += 0.10 * (pct_met / 6.0)                 # max +0.10
    ox += 0.05 * (pct_cys / 6.0)                 # max +0.05

    # Temperature — Arrhenius Q10 scaling
    # At 4°C → 0, at 25°C → small, at 50°C → significant
    temp_norm = max(0.0, (temp - 4) / 46.0)
    ox += 0.18 * (temp_norm ** 1.6)              # max +0.18 at 50°C

    # Protection
    if aa == "methionine":
        ox -= 0.14
    if "EDTA" in [buf, aa, salt]:
        ox -= 0.09
    if "ascorbic acid" in [buf, aa]:
        ox -= 0.07

    ox = float(np.clip(ox + np.random.normal(0, 0.015), 0, 1))

    # ── DEAMIDATION ───────────────────────────────────────
    # Target range: 0.03-0.15 optimal, 0.20-0.45 stressed
    deam = 0.02

    # Asn content (pct in PERCENT, normalize by max=8)
    deam += 0.22 * (pct_asn / 8.0)              # max +0.22

    # pH effect — cumulative but bounded
    if ph >= 7.4:
        deam += 0.15
    elif ph >= 7.0:
        deam += 0.10
    elif ph >= 6.5:
        deam += 0.04

    # Temperature
    temp_norm_d = max(0.0, (temp - 4) / 46.0)
    deam += 0.18 * (temp_norm_d ** 1.5)         # max +0.18 at 50°C

    # Buffer protection
    if buf in ["histidine", "citric acid"]:
        deam -= 0.07

    deam = float(np.clip(deam + np.random.normal(0, 0.015), 0, 1))

    # ── POTENCY ───────────────────────────────────────────
    potency = 1.0
    potency -= 0.42 * agg
    potency -= 0.30 * ox
    potency -= 0.20 * deam
    potency = float(np.clip(potency + np.random.normal(0, 0.015), 0, 1))

    # ── VISCOSITY ─────────────────────────────────────────
    visc = 0
    if protein_conc >= 100: visc += 1
    if sug_conc > 200:      visc += 1
    if aa_conc > 100:       visc += 1
    visc = int(np.clip(visc, 0, 2))

    # ── SHELF LIFE ────────────────────────────────────────
    shelf = 1.0 - (0.42 * agg + 0.31 * ox + 0.27 * deam)
    shelf = float(np.clip(shelf + np.random.normal(0, 0.018), 0, 1))

    # ── COMPOSITE SCORE ───────────────────────────────────
    composite = (
        (1 - agg)  * 0.40 +
        (1 - ox)   * 0.22 +
        (1 - deam) * 0.18 +
        potency    * 0.12 +
        shelf      * 0.08
    )
    composite = float(np.clip(composite, 0, 1))
    stable    = int(composite >= STABILITY_THRESHOLD)

    return {
        "aggregation_score":         round(agg,       4),
        "oxidation_level":           round(ox,        4),
        "deamidation_level":         round(deam,      4),
        "potency_retention":         round(potency,   4),
        "viscosity_class":           visc,
        "shelf_life_score":          round(shelf,     4),
        "composite_stability_score": round(composite, 4),
        "stable":                    stable,
    }


def make_row(protein_id, prow, buf, sug, sur, aa, salt,
             ph, temp, pconc):
    """Sample concentrations once and build complete row."""
    buf_conc  = sample_conc(buf)
    sug_conc  = sample_conc(sug)
    sur_conc  = sample_conc(sur)
    aa_conc   = sample_conc(aa)   if aa   else 0.0
    salt_conc = sample_conc(salt) if salt else 0.0

    outcomes = simulate_stability(
        prow, buf, sug, sur, aa, salt, ph, temp, pconc,
        buf_conc, sug_conc, sur_conc, aa_conc, salt_conc
    )

    return {
        "protein_id":        protein_id,
        "buffer":            buf,
        "sugar":             sug,
        "surfactant":        sur,
        "amino_acid":        aa   or "none",
        "salt":              salt or "none",
        "ph":                ph,
        "temperature_c":     temp,
        "protein_conc_mgmL": pconc,
        "buf_conc_mM":       buf_conc,
        "sug_conc_mM":       sug_conc,
        "sur_conc_mM":       sur_conc,
        "aa_conc_mM":        aa_conc,
        "salt_conc_mM":      salt_conc,
        **outcomes
    }


def generate_dataset():
    rows = []
    stress_templates = [
        {"temp": 50, "ph_near_pi": True,  "protein_conc": 100},
        {"temp": 40, "ph_near_pi": True,  "protein_conc": 100},
        {"temp": 50, "ph_near_pi": False, "protein_conc": 10},
        {"temp": 40, "ph_near_pi": False, "protein_conc": 100},
    ]

    for protein_id in proteins:
        prow = protein_df[protein_df["protein_id"] == protein_id].iloc[0]
        pi   = prow["isoelectric_point"]

        # Normal conditions
        for _ in range(N_NORMAL):
            rows.append(make_row(
                protein_id, prow,
                buf   = random.choice(buffers),
                sug   = random.choice(sugars),
                sur   = random.choice(surfactants),
                aa    = random.choice(amino_acids + [None]),
                salt  = random.choice(salts + [None]),
                ph    = random.choice(PH_VALUES),
                temp  = random.choice(TEMPERATURES),
                pconc = random.choice(PROTEIN_CONCS)
            ))

        # Deliberate stress
        for _ in range(N_STRESS):
            sc   = random.choice(stress_templates)
            temp = sc["temp"]
            pconc = sc["protein_conc"]
            if sc["ph_near_pi"]:
                ph = min(PH_VALUES, key=lambda x: abs(x - pi))
            else:
                ph = random.choice(PH_VALUES)

            rows.append(make_row(
                protein_id, prow,
                buf   = random.choice(buffers),
                sug   = random.choice(sugars),
                sur   = random.choice(surfactants),
                aa    = None,   # no amino acid stabilizer in stress
                salt  = random.choice(salts),
                ph    = ph,
                temp  = temp,
                pconc = pconc
            ))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating stability dataset v2...")
    print(f"  Proteins:     {len(proteins)}")
    print(f"  Rows/protein: {N_NORMAL + N_STRESS}")
    print(f"  Total:        {len(proteins) * (N_NORMAL + N_STRESS):,}\n")

    df = generate_dataset()
    df.to_csv("data/processed/stability_outcomes_v2.csv", index=False)

    print(f"Shape: {df.shape}")
    print(f"Null counts: {df.isnull().sum().sum()}")

    print(f"\nStable distribution:")
    print(df["stable"].value_counts())
    print(df["stable"].value_counts(normalize=True).round(3))

    print(f"\nMean outcomes:")
    print(df[["aggregation_score","oxidation_level",
              "deamidation_level","potency_retention",
              "composite_stability_score"]].mean().round(3))

    print(f"\nOutcome ranges:")
    print(df[["aggregation_score","oxidation_level",
              "deamidation_level","composite_stability_score"]].describe().round(3))

    print(f"\nTemp distribution:")
    print(df["temperature_c"].value_counts().sort_index())

    print(f"\nProtein conc distribution:")
    print(df["protein_conc_mgmL"].value_counts().sort_index())

    print(f"\nStable rate by protein type:")
    merged = df.merge(protein_df[["protein_id","query_label"]], on="protein_id")
    print(merged.groupby("query_label")["stable"].mean().round(3))

    print(f"\nSaved to data/processed/stability_outcomes_v2.csv")
# This won't run — just showing the patch location
