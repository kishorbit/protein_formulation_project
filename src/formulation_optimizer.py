import pandas as pd
import numpy as np
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

# ── Load models & feature schema ──────────────────────────
cls_model    = joblib.load("models/xgb_classifier.pkl")
reg_model    = joblib.load("models/lgb_regressor.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

pf           = pd.read_csv("data/processed/protein_features_expanded.csv")
excipient_df = pd.read_csv("data/processed/excipient_db.csv")

# ── Excipient search space ────────────────────────────────
BUFFERS     = excipient_df[excipient_df["class"]=="buffer"]["name"].tolist()
SUGARS      = excipient_df[excipient_df["class"]=="sugar"]["name"].tolist()
SURFACTANTS = excipient_df[excipient_df["class"]=="surfactant"]["name"].tolist()
AMINO_ACIDS = excipient_df[excipient_df["class"]=="amino_acid"]["name"].tolist() + ["none"]
SALTS       = excipient_df[excipient_df["class"]=="salt"]["name"].tolist() + ["none"]
PH_VALUES   = [5.0, 5.5, 6.0, 6.5, 7.0, 7.4]
TEMPERATURES = [4, 25]          # optimizer targets storage conditions only
PROTEIN_CONCS = [1, 10, 100]

def get_mid_conc(name):
    """Return midpoint of allowed concentration range."""
    if name == "none" or name is None:
        return 0.0
    row = excipient_df[excipient_df["name"] == name]
    if row.empty:
        return 0.0
    return float((row["conc_min_mM"].values[0] + row["conc_max_mM"].values[0]) / 2)

def get_conc_range(name, steps=3):
    """Return evenly spaced concentrations across allowed range."""
    if name == "none" or name is None:
        return [0.0]
    row = excipient_df[excipient_df["name"] == name]
    if row.empty:
        return [0.0]
    lo = float(row["conc_min_mM"].values[0])
    hi = float(row["conc_max_mM"].values[0])
    return np.linspace(lo, hi, steps).tolist()

def build_feature_vector(prow, buf, sug, sur, aa, salt,
                          ph, temp, pconc,
                          buf_conc, sug_conc, sur_conc,
                          aa_conc, salt_conc):
    """Build one feature vector matching training schema."""

    # Raw features
    raw = {
        "ph":                   ph,
        "temperature_c":        temp,
        "protein_conc_mgmL":    pconc,
        "buf_conc_mM":          buf_conc,
        "sug_conc_mM":          sug_conc,
        "sur_conc_mM":          sur_conc,
        "aa_conc_mM":           aa_conc,
        "salt_conc_mM":         salt_conc,
        "isoelectric_point":    prow["isoelectric_point"],
        "instability_index":    prow["instability_index"],
        "gravy_score":          prow["gravy_score"],
        "pct_asn":              prow["pct_asn"],
        "pct_met":              prow["pct_met"],
        "pct_cys":              prow["pct_cys"],
        "pct_trp":              prow["pct_trp"],
        "met_exposed_fraction": prow["met_exposed_fraction"],
        "ox_mean_rsa":          prow["ox_mean_rsa"],
        "agg_hotspot_frac":     prow["agg_hotspot_frac"]
                                if pd.notna(prow["agg_hotspot_frac"])
                                else pf["agg_hotspot_frac"].median(),
        "agg_mean":             prow["agg_mean"],
        "ox_risk_composite":    prow["ox_risk_composite"],
        # Interaction features
        "ph_pi_dist":           abs(ph - prow["isoelectric_point"]),
        "temp_conc":            temp * pconc,
        "instab_gravy":         prow["instability_index"] * max(0, prow["gravy_score"]),
        "met_temp":             prow["met_exposed_fraction"] * temp,
    }

    # Categorical one-hot — must match training columns exactly
    cat_map = {
        "buffer":      buf,
        "sugar":       sug,
        "surfactant":  sur,
        "amino_acid":  aa,
        "salt":        salt,
        "query_label": prow["query_label"],
    }

    vec = {col: 0.0 for col in feature_cols}
    for col, val in raw.items():
        if col in vec:
            vec[col] = float(val)
    for prefix, val in cat_map.items():
        key = f"{prefix}_{val}"
        if key in vec:
            vec[key] = 1.0

    return np.array([vec[col] for col in feature_cols], dtype=np.float32)


def optimize_formulation(protein_id, target_temp=25, target_conc=10,
                          top_n=10, verbose=True):
    """
    Grid search over excipient combinations for a given protein.
    Returns top_n formulations ranked by composite stability score.
    """
    if protein_id not in pf["protein_id"].values:
        raise ValueError(f"Protein {protein_id} not found.")

    prow = pf[pf["protein_id"] == protein_id].iloc[0]

    if verbose:
        print(f"\n{'='*65}")
        print(f"Optimizing formulation for: {protein_id}")
        print(f"  Protein type:  {prow['query_label']}")
        print(f"  pI:            {prow['isoelectric_point']:.2f}")
        print(f"  Instability:   {prow['instability_index']:.1f}")
        print(f"  GRAVY:         {prow['gravy_score']:.3f}")
        print(f"  Met exposed:   {prow['met_exposed_fraction']:.3f}")
        print(f"  Target temp:   {target_temp}°C")
        print(f"  Target conc:   {target_conc} mg/mL")
        print(f"{'='*65}")

    results = []

    # Grid: buffers × sugars × surfactants × amino_acids × salts × pH
    # Use mid-conc for speed; top candidates get full conc sweep
    combos = list(itertools.product(
        BUFFERS, SUGARS, SURFACTANTS, AMINO_ACIDS, SALTS, PH_VALUES
    ))

    if verbose:
        print(f"Evaluating {len(combos):,} formulation combinations...")

    X_batch = []
    meta    = []

    for buf, sug, sur, aa, salt, ph in combos:
        buf_conc  = get_mid_conc(buf)
        sug_conc  = get_mid_conc(sug)
        sur_conc  = get_mid_conc(sur)
        aa_conc   = get_mid_conc(aa)
        salt_conc = get_mid_conc(salt)

        vec = build_feature_vector(
            prow, buf, sug, sur, aa, salt,
            ph, target_temp, target_conc,
            buf_conc, sug_conc, sur_conc, aa_conc, salt_conc
        )
        X_batch.append(vec)
        meta.append({
            "buffer": buf, "sugar": sug, "surfactant": sur,
            "amino_acid": aa, "salt": salt, "ph": ph,
            "buf_conc_mM": buf_conc, "sug_conc_mM": sug_conc,
            "sur_conc_mM": sur_conc, "aa_conc_mM": aa_conc,
            "salt_conc_mM": salt_conc,
        })

    X_batch = np.array(X_batch, dtype=np.float32)

    # Predict
    proba     = cls_model.predict_proba(X_batch)[:, 1]
    composite = reg_model.predict(X_batch)

    for i, m in enumerate(meta):
        m["pred_stable_proba"]    = round(float(proba[i]),     4)
        m["pred_composite_score"] = round(float(composite[i]), 4)

    results_df = pd.DataFrame(meta)
    results_df = results_df.sort_values(
        ["pred_composite_score", "pred_stable_proba"],
        ascending=False
    ).reset_index(drop=True)

    top = results_df.head(top_n).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top.insert(1, "protein_id", protein_id)

    if verbose:
        print(f"\nTop {top_n} formulations:")
        print(top[["rank","buffer","sugar","surfactant","amino_acid","salt",
                   "ph","pred_stable_proba","pred_composite_score"]].to_string(index=False))

    return top, results_df


def batch_optimize(protein_ids, target_temp=25, target_conc=10, top_n=5):
    """Optimize multiple proteins, return combined results."""
    all_results = []
    for pid in protein_ids:
        try:
            top, _ = optimize_formulation(pid, target_temp, target_conc,
                                          top_n=top_n, verbose=False)
            all_results.append(top)
            print(f"  {pid}: best composite={top.iloc[0]['pred_composite_score']:.4f}  "
                  f"pH={top.iloc[0]['ph']}  "
                  f"buf={top.iloc[0]['buffer']}  "
                  f"aa={top.iloc[0]['amino_acid']}")
        except Exception as e:
            print(f"  {pid}: ERROR — {e}")

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


# ── Demo run ──────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("outputs/optimization", exist_ok=True)

    # 1. Single protein — detailed
    sample_protein = pf["protein_id"].iloc[0]
    top10, full = optimize_formulation(
        sample_protein, target_temp=25, target_conc=10, top_n=10
    )
    top10.to_csv(f"outputs/optimization/{sample_protein}_top10.csv", index=False)

    # 2. Batch — one representative per protein type
    print(f"\n{'='*65}")
    print("BATCH OPTIMIZATION — one protein per type")
    print(f"{'='*65}")
    representatives = (
        pf.groupby("query_label")["protein_id"].first().values
    )
    batch_df = batch_optimize(representatives, target_temp=25, target_conc=10, top_n=5)
    batch_df.to_csv("outputs/optimization/batch_top5_per_type.csv", index=False)

    # 3. Worst-case stress test — 37°C, 100 mg/mL
    print(f"\n{'='*65}")
    print("STRESS TEST — 37°C, 100 mg/mL")
    print(f"{'='*65}")
    stress_top, _ = optimize_formulation(
        sample_protein, target_temp=37, target_conc=100, top_n=5
    )
    stress_top.to_csv(f"outputs/optimization/{sample_protein}_stress_top5.csv", index=False)

    print(f"\nSaved to outputs/optimization/")
