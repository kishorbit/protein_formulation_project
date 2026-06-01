"""
reliability_gate.py
-------------------
Single entry point for prediction reliability at inference time.

Usage:
    from src.reliability_gate import assess_reliability
    result = assess_reliability(protein_features_dict, predictions_dict)
    print(result['reliability_label'])   # 'high' | 'medium' | 'low'
    print(result['warnings'])            # list of strings
"""
import numpy as np
import pandas as pd
import joblib
import os

# ── Thresholds (validated against held-out proteins) ─────
UNCERTAINTY_THRESHOLD = 0.025   # precision=1.00 recall=0.67
OOD_CONTAMINATION     = 0.15    # hybrid Mahalanobis + sigma flags

# Per-target reliability based on held-out validation
TARGET_RELIABILITY = {
    "aggregation_score":  {"mean_r2": 0.909, "min_r2": 0.649, "note": "robust"},
    "oxidation_level":    {"mean_r2": 0.594, "min_r2": -10.3, "note": "unreliable on Cys/Met outliers"},
    "deamidation_level":  {"mean_r2": 0.759, "min_r2": -0.41, "note": "unreliable on high-Asn proteins"},
    "potency_retention":  {"mean_r2": 0.848, "min_r2": -0.35, "note": "generally reliable"},
    "shelf_life_score":   {"mean_r2": 0.785, "min_r2": -0.82, "note": "moderate reliability"},
}

# OOD feature thresholds (mean ± 2σ from training set)
OOD_FEATURES = ["pct_cys","pct_met","pct_asn","instability_index",
                 "sequence_length","gravy_score","agg_mean","agg_hotspot_frac"]


def load_ood_detector(model_dir="outputs/models"):
    path = os.path.join(model_dir, "ood_detector.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def assess_reliability(protein_features: dict,
                       predictions: dict,
                       model_dir: str = "outputs/models") -> dict:
    """
    Assess reliability of predictions for a single protein.

    Parameters
    ----------
    protein_features : dict
        Protein-level features, e.g. {"pct_cys": 2.1, "pct_met": 1.8, ...}
    predictions : dict
        Model outputs, must include "mean_uncertainty" and optionally
        per-target pred values.
    model_dir : str
        Directory containing ood_detector.pkl

    Returns
    -------
    dict with keys:
        reliability_label : 'high' | 'medium' | 'low'
        uncertainty_flag  : bool
        ood_flag          : bool
        sigma_flags       : dict  feature -> n_sigma
        warnings          : list[str]
        target_warnings   : dict  target -> warning string
        details           : dict  raw scores
    """
    warnings_out      = []
    target_warnings   = {}
    uncertainty_flag  = False
    ood_flag          = False
    sigma_flags       = {}

    # ── 1. Uncertainty gate (primary) ────────────────────
    mean_unc = predictions.get("mean_uncertainty", None)
    if mean_unc is not None:
        if mean_unc >= UNCERTAINTY_THRESHOLD:
            uncertainty_flag = True
            warnings_out.append(
                f"High prediction uncertainty ({mean_unc:.4f} >= {UNCERTAINTY_THRESHOLD}). "
                f"Regression predictions may be unreliable."
            )

    # ── 2. OOD detector (secondary) ──────────────────────
    ood_data = load_ood_detector(model_dir)
    if ood_data is not None:
        feats    = ood_data["features"]
        scaler   = ood_data["scaler"]
        detector = ood_data["detector"]
        thresh   = ood_data.get("thresholds", {})

        feat_vec = np.array([[protein_features.get(f, np.nan) for f in feats]])

        # fill missing with median if needed
        for i, f in enumerate(feats):
            if np.isnan(feat_vec[0, i]):
                feat_vec[0, i] = thresh.get(f, {}).get("mean", 0)

        X_scaled  = scaler.transform(feat_vec)
        mahal_pred = detector.predict(X_scaled)[0]   # -1 = outlier
        mahal_score = detector.score_samples(X_scaled)[0]

        # per-feature sigma check
        n_sigma_flags = 0
        for f in feats:
            if f not in thresh:
                continue
            v  = protein_features.get(f, None)
            if v is None:
                continue
            hi = thresh[f]["hi"]
            lo = thresh[f]["lo"]
            mu = thresh[f]["mean"]
            sd = thresh[f]["std"]
            if v > hi or v < lo:
                n_sigma = abs(v - mu) / sd if sd > 0 else 0
                sigma_flags[f] = round(n_sigma, 1)
                n_sigma_flags += 1

        ood_flag = (mahal_pred == -1) or (n_sigma_flags >= 1)

        if ood_flag:
            flagged_feats = list(sigma_flags.keys())
            if mahal_pred == -1:
                warnings_out.append(
                    f"Protein is out-of-distribution (Mahalanobis score={mahal_score:.1f}). "
                    f"Model has limited training data for proteins like this."
                )
            if flagged_feats:
                sigma_str = ", ".join(
                    f"{f}={protein_features.get(f,'?'):.2f} ({sigma_flags[f]:.1f}σ)"
                    for f in flagged_feats
                )
                warnings_out.append(f"Outlier composition features: {sigma_str}")

            # per-target warnings based on which features are outlying
            if "pct_cys" in sigma_flags or "pct_met" in sigma_flags:
                target_warnings["oxidation_level"] = (
                    "Unreliable — protein has outlier Cys/Met content. "
                    "oxidation_level R² collapses on these proteins."
                )
            if "pct_asn" in sigma_flags:
                target_warnings["deamidation_level"] = (
                    "Unreliable — protein has outlier Asn content. "
                    "deamidation_level R² collapses on high-Asn proteins."
                )
            if "sequence_length" in sigma_flags or "instability_index" in sigma_flags:
                for t in ["shelf_life_score", "potency_retention"]:
                    target_warnings[t] = (
                        f"Reduced reliability — outlier sequence length or instability. "
                        f"{TARGET_RELIABILITY[t]['note']}."
                    )
    else:
        warnings_out.append("OOD detector not found — skipping OOD check.")

    # ── 3. Assign reliability label ──────────────────────
    if uncertainty_flag:
        reliability_label = "low"
    elif ood_flag:
        reliability_label = "medium"
    else:
        reliability_label = "high"

    return {
        "reliability_label": reliability_label,
        "uncertainty_flag":  uncertainty_flag,
        "ood_flag":          ood_flag,
        "sigma_flags":       sigma_flags,
        "warnings":          warnings_out,
        "target_warnings":   target_warnings,
        "details": {
            "mean_uncertainty": mean_unc,
            "ood_threshold":    UNCERTAINTY_THRESHOLD,
        }
    }


def batch_assess(df_proteins: pd.DataFrame,
                 df_predictions: pd.DataFrame,
                 protein_id_col: str = "protein_id",
                 model_dir: str = "outputs/models") -> pd.DataFrame:
    """
    Run assess_reliability over a DataFrame of proteins.

    Parameters
    ----------
    df_proteins   : DataFrame with protein feature columns
    df_predictions: DataFrame with prediction columns incl. mean_uncertainty
    protein_id_col: join key

    Returns
    -------
    DataFrame with reliability_label, uncertainty_flag, ood_flag,
    n_sigma_flags, warnings columns added.
    """
    merged = df_proteins.merge(df_predictions, on=protein_id_col, how="inner")
    records = []
    for _, row in merged.iterrows():
        prot_feat = row.to_dict()
        preds     = row.to_dict()
        result    = assess_reliability(prot_feat, preds, model_dir)
        records.append({
            protein_id_col:      row[protein_id_col],
            "reliability_label": result["reliability_label"],
            "uncertainty_flag":  result["uncertainty_flag"],
            "ood_flag":          result["ood_flag"],
            "n_sigma_flags":     len(result["sigma_flags"]),
            "warnings":          " | ".join(result["warnings"]),
            "target_warnings":   str(result["target_warnings"]),
        })
    return pd.DataFrame(records)


# ── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    print("\nReliability gate — smoke test")
    print("="*55)

    test_cases = [
        {
            "label": "Normal protein, low uncertainty",
            "features": {"pct_cys": 2.1, "pct_met": 1.8, "pct_asn": 3.2,
                         "instability_index": 42.0, "sequence_length": 280,
                         "gravy_score": -0.2, "agg_mean": -0.1,
                         "agg_hotspot_frac": 0.05},
            "predictions": {"mean_uncertainty": 0.018},
        },
        {
            "label": "High Cys protein (A6BM72-like), low uncertainty",
            "features": {"pct_cys": 14.0, "pct_met": 1.4, "pct_asn": 3.6,
                         "instability_index": 59.0, "sequence_length": 1044,
                         "gravy_score": -0.29, "agg_mean": -0.19,
                         "agg_hotspot_frac": 0.02},
            "predictions": {"mean_uncertainty": 0.019},
        },
        {
            "label": "Normal protein, high uncertainty (FN case)",
            "features": {"pct_cys": 2.0, "pct_met": 2.1, "pct_asn": 3.0,
                         "instability_index": 44.0, "sequence_length": 220,
                         "gravy_score": -0.15, "agg_mean": -0.12,
                         "agg_hotspot_frac": 0.06},
            "predictions": {"mean_uncertainty": 0.032},
        },
        {
            "label": "High Met protein (P01570-like)",
            "features": {"pct_cys": 3.2, "pct_met": 6.3, "pct_asn": 4.8,
                         "instability_index": 61.0, "sequence_length": 189,
                         "gravy_score": -0.22, "agg_mean": -0.11,
                         "agg_hotspot_frac": 0.04},
            "predictions": {"mean_uncertainty": 0.021},
        },
    ]

    for tc in test_cases:
        result = assess_reliability(tc["features"], tc["predictions"])
        print(f"\n  [{tc['label']}]")
        print(f"    reliability : {result['reliability_label'].upper()}")
        print(f"    unc_flag    : {result['uncertainty_flag']}  "
              f"ood_flag: {result['ood_flag']}  "
              f"sigma_flags: {result['sigma_flags']}")
        for w in result["warnings"]:
            print(f"    WARNING: {w}")
        for t, w in result["target_warnings"].items():
            print(f"    TARGET  {t}: {w}")

    print()
    print("  Reliability thresholds:")
    print(f"    uncertainty >= {UNCERTAINTY_THRESHOLD} -> LOW reliability")
    print(f"    OOD flag (Mahalanobis or sigma) -> MEDIUM reliability")
    print(f"    neither -> HIGH reliability")
    print()
    print("  Per-target held-out R2 (reference):")
    for t, info in TARGET_RELIABILITY.items():
        print(f"    {t:<25} mean_r2={info['mean_r2']:.3f}  "
              f"note={info['note']}")
    print("="*55)
