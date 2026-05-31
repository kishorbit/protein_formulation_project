"""
Protein Formulation Stability — Prediction API
==============================================
Two entry points:
  predict_single()  — one formulation, instant result
  predict_batch()   — dataframe of formulations, bulk scoring
  recommend()       — given a protein, return ranked formulations
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Load once at import ────────────────────────────────────
_cls_model    = joblib.load("models/xgb_classifier.pkl")
_reg_model    = joblib.load("models/lgb_regressor.pkl")
_feature_cols = joblib.load("models/feature_cols.pkl")
_pf           = pd.read_csv("data/processed/protein_features_expanded.csv")
_excipient_df = pd.read_csv("data/processed/excipient_db.csv")

_pf["agg_hotspot_frac"] = _pf["agg_hotspot_frac"].fillna(
    _pf["agg_hotspot_frac"].median()
)

# ── Allowed values (for validation) ───────────────────────
ALLOWED = {
    "buffer":     _excipient_df[_excipient_df["class"]=="buffer"]["name"].tolist(),
    "sugar":      _excipient_df[_excipient_df["class"]=="sugar"]["name"].tolist(),
    "surfactant": _excipient_df[_excipient_df["class"]=="surfactant"]["name"].tolist(),
    "amino_acid": _excipient_df[_excipient_df["class"]=="amino_acid"]["name"].tolist() + ["none"],
    "salt":       _excipient_df[_excipient_df["class"]=="salt"]["name"].tolist() + ["none"],
    "ph":         [5.0, 5.5, 6.0, 6.5, 7.0, 7.4],
    "temperature_c":     [4, 25, 37, 40, 50],
    "protein_conc_mgmL": [1, 10, 100],
}


def _get_conc(name, conc_override=None):
    if name == "none" or name is None:
        return 0.0
    if conc_override is not None:
        return float(conc_override)
    row = _excipient_df[_excipient_df["name"] == name]
    if row.empty:
        return 0.0
    return float((row["conc_min_mM"].values[0] + row["conc_max_mM"].values[0]) / 2)


def _validate(protein_id, formulation):
    errors = []
    if protein_id not in _pf["protein_id"].values:
        errors.append(f"protein_id '{protein_id}' not found.")
    for field, allowed in ALLOWED.items():
        if field in formulation and formulation[field] not in allowed:
            errors.append(f"'{field}={formulation[field]}' not in allowed: {allowed}")
    if errors:
        raise ValueError("\n".join(errors))


def _build_vector(prow, formulation):
    buf   = formulation["buffer"]
    sug   = formulation["sugar"]
    sur   = formulation["surfactant"]
    aa    = formulation.get("amino_acid", "none")
    salt  = formulation.get("salt", "none")
    ph    = float(formulation["ph"])
    temp  = float(formulation["temperature_c"])
    pconc = float(formulation["protein_conc_mgmL"])

    buf_conc  = _get_conc(buf,  formulation.get("buf_conc_mM"))
    sug_conc  = _get_conc(sug,  formulation.get("sug_conc_mM"))
    sur_conc  = _get_conc(sur,  formulation.get("sur_conc_mM"))
    aa_conc   = _get_conc(aa,   formulation.get("aa_conc_mM"))
    salt_conc = _get_conc(salt, formulation.get("salt_conc_mM"))

    pi = float(prow["isoelectric_point"])

    raw = {
        "ph":                   ph,
        "temperature_c":        temp,
        "protein_conc_mgmL":    pconc,
        "buf_conc_mM":          buf_conc,
        "sug_conc_mM":          sug_conc,
        "sur_conc_mM":          sur_conc,
        "aa_conc_mM":           aa_conc,
        "salt_conc_mM":         salt_conc,
        "isoelectric_point":    pi,
        "instability_index":    float(prow["instability_index"]),
        "gravy_score":          float(prow["gravy_score"]),
        "pct_asn":              float(prow["pct_asn"]),
        "pct_met":              float(prow["pct_met"]),
        "pct_cys":              float(prow["pct_cys"]),
        "pct_trp":              float(prow["pct_trp"]),
        "met_exposed_fraction": float(prow["met_exposed_fraction"]),
        "ox_mean_rsa":          float(prow["ox_mean_rsa"]),
        "agg_hotspot_frac":     float(prow["agg_hotspot_frac"]),
        "agg_mean":             float(prow["agg_mean"]),
        "ox_risk_composite":    float(prow["ox_risk_composite"]),
        "ph_pi_dist":           abs(ph - pi),
        "temp_conc":            temp * pconc,
        "instab_gravy":         float(prow["instability_index"]) * max(0, float(prow["gravy_score"])),
        "met_temp":             float(prow["met_exposed_fraction"]) * temp,
    }

    cat_map = {
        "buffer":      buf,
        "sugar":       sug,
        "surfactant":  sur,
        "amino_acid":  aa,
        "salt":        salt,
        "query_label": prow["query_label"],
    }

    vec = {col: 0.0 for col in _feature_cols}
    for col, val in raw.items():
        if col in vec:
            vec[col] = float(val)
    for prefix, val in cat_map.items():
        key = f"{prefix}_{val}"
        if key in vec:
            vec[key] = 1.0

    return np.array([vec[col] for col in _feature_cols], dtype=np.float32)


# ── Public API ─────────────────────────────────────────────

def predict_single(protein_id: str, formulation: dict) -> dict:
    """
    Score one formulation for a given protein.

    Parameters
    ----------
    protein_id  : UniProt ID (e.g. 'O15520')
    formulation : dict with keys:
        required — buffer, sugar, surfactant, ph,
                   temperature_c, protein_conc_mgmL
        optional — amino_acid, salt,
                   buf_conc_mM, sug_conc_mM, sur_conc_mM,
                   aa_conc_mM, salt_conc_mM

    Returns
    -------
    dict with prediction results + input echo
    """
    _validate(protein_id, formulation)
    prow = _pf[_pf["protein_id"] == protein_id].iloc[0]
    vec  = _build_vector(prow, formulation).reshape(1, -1)

    proba     = float(_cls_model.predict_proba(vec)[0, 1])
    composite = float(_reg_model.predict(vec)[0])
    stable    = int(proba >= 0.5)

    # Risk flags
    flags = []
    if formulation.get("temperature_c", 25) >= 40:
        flags.append("HIGH_TEMP")
    if formulation.get("protein_conc_mgmL", 10) >= 100:
        flags.append("HIGH_CONC")
    if abs(float(formulation["ph"]) - float(prow["isoelectric_point"])) < 1.0:
        flags.append("NEAR_PI")
    if formulation.get("amino_acid", "none") == "none":
        flags.append("NO_AA_STABILIZER")

    return {
        "protein_id":          protein_id,
        "protein_type":        prow["query_label"],
        "pI":                  round(float(prow["isoelectric_point"]), 2),
        "formulation":         formulation,
        "pred_stable":         stable,
        "pred_stable_proba":   round(proba, 4),
        "pred_composite_score": round(composite, 4),
        "stability_grade":     "A" if composite >= 0.80 else
                               "B" if composite >= 0.70 else
                               "C" if composite >= 0.60 else
                               "D" if composite >= 0.55 else "F",
        "risk_flags":          flags,
    }


def predict_batch(protein_id: str, formulations: pd.DataFrame) -> pd.DataFrame:
    """
    Score multiple formulations for one protein.

    Parameters
    ----------
    protein_id   : UniProt ID
    formulations : DataFrame where each row is a formulation

    Returns
    -------
    DataFrame with predictions appended
    """
    prow   = _pf[_pf["protein_id"] == protein_id].iloc[0]
    vecs   = np.array([
        _build_vector(prow, row.to_dict())
        for _, row in formulations.iterrows()
    ], dtype=np.float32)

    probas     = _cls_model.predict_proba(vecs)[:, 1]
    composites = _reg_model.predict(vecs)

    out = formulations.copy()
    out.insert(0, "protein_id", protein_id)
    out["pred_stable_proba"]    = probas.round(4)
    out["pred_composite_score"] = composites.round(4)
    out["pred_stable"]          = (probas >= 0.5).astype(int)
    out["stability_grade"]      = pd.cut(
        composites,
        bins=[0, 0.55, 0.60, 0.70, 0.80, 1.01],
        labels=["F","D","C","B","A"]
    )
    return out.sort_values("pred_composite_score", ascending=False).reset_index(drop=True)


def recommend(protein_id: str,
              temperature_c: int = 25,
              protein_conc_mgmL: int = 10,
              top_n: int = 5,
              require_amino_acid: bool = True) -> pd.DataFrame:
    """
    Return top_n recommended formulations for a protein.
    Thin wrapper around the optimizer using mid-point concentrations.
    """
    import itertools
    from src.formulation_optimizer import optimize_formulation
    top, _ = optimize_formulation(
        protein_id,
        target_temp=temperature_c,
        target_conc=protein_conc_mgmL,
        top_n=top_n,
        verbose=False
    )
    if require_amino_acid:
        top = top[top["amino_acid"] != "none"].head(top_n)
    return top


# ── Quick self-test ────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("SELF-TEST — predict_single()")
    print("="*55)

    test_cases = [
        # Best-case formulation
        {
            "protein_id": "O15520",
            "formulation": {
                "buffer": "histidine", "sugar": "sucrose",
                "surfactant": "polysorbate 80", "amino_acid": "methionine",
                "salt": "none", "ph": 5.5,
                "temperature_c": 25, "protein_conc_mgmL": 10,
            }
        },
        # Stress — high temp, high conc, near pI
        {
            "protein_id": "O15520",
            "formulation": {
                "buffer": "phosphoric acid", "sugar": "mannitol",
                "surfactant": "poloxamer 188", "amino_acid": "none",
                "salt": "sodium chloride", "ph": 7.0,
                "temperature_c": 40, "protein_conc_mgmL": 100,
            }
        },
        # Different protein type
        {
            "protein_id": "P00740",
            "formulation": {
                "buffer": "histidine", "sugar": "sucrose",
                "surfactant": "polysorbate 80", "amino_acid": "arginine",
                "salt": "none", "ph": 7.0,
                "temperature_c": 25, "protein_conc_mgmL": 10,
            }
        },
    ]

    for tc in test_cases:
        result = predict_single(tc["protein_id"], tc["formulation"])
        print(f"\nProtein : {result['protein_id']}  ({result['protein_type']})")
        print(f"pI      : {result['pI']}")
        print(f"Grade   : {result['stability_grade']}  "
              f"(composite={result['pred_composite_score']}  "
              f"p_stable={result['pred_stable_proba']})")
        print(f"Flags   : {result['risk_flags'] or 'none'}")

    print()
    print("="*55)
    print("SELF-TEST — predict_batch()")
    print("="*55)

    batch_input = pd.DataFrame([
        {"buffer":"histidine","sugar":"sucrose","surfactant":"polysorbate 80",
         "amino_acid":"methionine","salt":"none","ph":ph,
         "temperature_c":25,"protein_conc_mgmL":10}
        for ph in [5.0, 5.5, 6.0, 6.5, 7.0, 7.4]
    ])
    batch_out = predict_batch("O15520", batch_input)
    print(batch_out[["ph","pred_stable_proba",
                      "pred_composite_score","stability_grade"]].to_string(index=False))

    print("\nAPI ready.")
