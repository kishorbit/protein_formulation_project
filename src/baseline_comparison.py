import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              mean_squared_error, r2_score)
from xgboost import XGBRegressor
import lightgbm as lgb

COMPOSITE_WEIGHTS = {
    "aggregation_score":  0.35,
    "oxidation_level":    0.25,
    "deamidation_level":  0.20,
    "potency_retention":  0.15,
    "shelf_life_score":   0.05,
}
STABILITY_THRESHOLD = 0.40
REGRESSION_TARGETS  = [
    "aggregation_score","oxidation_level","deamidation_level",
    "potency_retention","shelf_life_score",
]

NAMED_CONSTRAINTS = {
    "aggregation_score": {
        "temperature_c":+1,"high_temp_flag":+1,"instability_index":+1,
        "agg_mean":+1,"agg_hotspot_frac":+1,"sug_conc_mM":-1,
        "sur_conc_mM":-1,"sug_x_instability":-1,
    },
    "oxidation_level": {
        "temperature_c":+1,"high_temp_flag":+1,"pct_met":+1,"pct_trp":+1,
    },
    "deamidation_level": {
        "temperature_c":+1,"high_temp_flag":+1,"pct_asn":+1,"ph":+1,
    },
    "potency_retention": {
        "temperature_c":-1,"high_temp_flag":-1,"instability_index":-1,
        "agg_mean":-1,"sug_conc_mM":+1,
    },
    "shelf_life_score": {
        "temperature_c":-1,"high_temp_flag":-1,"instability_index":-1,
        "sug_conc_mM":+1,"sur_conc_mM":+1,
    },
}

def build_constraint_tuple(target, feature_cols):
    c = NAMED_CONSTRAINTS.get(target, {})
    return tuple(c.get(f, 0) for f in feature_cols)

def derive_composite(y_pred):
    return (
        COMPOSITE_WEIGHTS["aggregation_score"]  * y_pred[:, 0]
      + COMPOSITE_WEIGHTS["oxidation_level"]    * y_pred[:, 1]
      + COMPOSITE_WEIGHTS["deamidation_level"]  * y_pred[:, 2]
      + COMPOSITE_WEIGHTS["potency_retention"]  * (1 - y_pred[:, 3])
      + COMPOSITE_WEIGHTS["shelf_life_score"]   * (1 - y_pred[:, 4])
    )

# ── Load data ─────────────────────────────────────────────
print("\nLoading data...")
df = pd.read_csv("data/processed/dataset_merged.csv")
feature_cols = pd.read_csv(
    "data/processed/feature_cols.csv", header=None)[0].tolist()
feature_cols = [c for c in feature_cols if c in df.columns]

proteins = df["protein_id"].unique()
print(f"  Proteins: {len(proteins)}  |  Features: {len(feature_cols)}")

# ── Models to benchmark ───────────────────────────────────
def make_models(feature_cols):
    return {
        "Majority class (dummy)": {
            "type": "dummy_reg",
        },
        "Mean predictor (dummy)": {
            "type": "mean_reg",
        },
        "Ridge regression": {
            "type": "multi_reg",
            "model": MultiOutputRegressor(Ridge(alpha=1.0)),
        },
        "Random Forest": {
            "type": "multi_reg",
            "model": MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=100, max_depth=6,
                    random_state=42, n_jobs=-1)),
        },
        "LightGBM": {
            "type": "multi_reg",
            "model": MultiOutputRegressor(
                lgb.LGBMRegressor(
                    n_estimators=200, max_depth=4,
                    learning_rate=0.05, random_state=42,
                    verbose=-1)),
        },
        "XGBoost (our model)": {
            "type": "xgb_monotone",
        },
    }

# ── LOO loop ──────────────────────────────────────────────
print("\nRunning LOO comparison across all models...")
print(f"{'Model':<30} {'ROC-AUC':>8} {'PR-AUC':>8} "
      f"{'Comp R²':>8} {'Agg R²':>8}")
print("-" * 70)

results = []

for model_name, model_cfg in make_models(feature_cols).items():
    roc_aucs, pr_aucs, comp_r2s, agg_r2s = [], [], [], []

    for held_out in proteins:
        train_mask = df["protein_id"] != held_out
        test_mask  = df["protein_id"] == held_out

        X_tr_raw = df.loc[train_mask, feature_cols].values
        X_te_raw = df.loc[test_mask,  feature_cols].values
        y_tr     = df.loc[train_mask, REGRESSION_TARGETS].values
        y_te     = df.loc[test_mask,  REGRESSION_TARGETS].values
        y_bin    = df.loc[test_mask,  "stable"].values
        y_comp_true = df.loc[test_mask, "composite_stability_score"].values

        imp = KNNImputer(n_neighbors=5)
        X_tr = imp.fit_transform(X_tr_raw)
        X_te = imp.transform(X_te_raw)
        scl = MinMaxScaler()
        X_tr = scl.fit_transform(X_tr)
        X_te = scl.transform(X_te)

        # ── Predict ───────────────────────────────────────
        if model_cfg["type"] == "dummy_reg":
            # Majority class: predict training mean for all targets
            y_pred = np.tile(y_tr.mean(axis=0), (len(X_te), 1))

        elif model_cfg["type"] == "mean_reg":
            # Mean predictor: per-column mean
            y_pred = np.tile(y_tr.mean(axis=0), (len(X_te), 1))
            # (same as dummy here — included as explicit baseline)
            y_pred = y_pred + np.random.normal(0, 1e-6, y_pred.shape)

        elif model_cfg["type"] == "multi_reg":
            m = model_cfg["model"]
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)

        elif model_cfg["type"] == "xgb_monotone":
            estimators = []
            for i, target in enumerate(REGRESSION_TARGETS):
                xgb = XGBRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=0, tree_method="hist",
                    monotone_constraints=build_constraint_tuple(
                        target, feature_cols),
                )
                xgb.fit(X_tr, y_tr[:, i])
                estimators.append(xgb)
            y_pred = np.column_stack([e.predict(X_te) for e in estimators])

        # ── Metrics ───────────────────────────────────────
        comp_pred = derive_composite(y_pred)
        signal    = 1 - comp_pred

        if len(np.unique(y_bin)) > 1:
            roc_aucs.append(roc_auc_score(y_bin, signal))
            pr_aucs.append(average_precision_score(y_bin, signal))

        comp_r2s.append(r2_score(y_comp_true, comp_pred))
        agg_r2s.append(r2_score(y_te[:, 0], y_pred[:, 0]))

    mean_roc = np.nanmean(roc_aucs)
    mean_pr  = np.nanmean(pr_aucs)
    mean_cr2 = np.mean(comp_r2s)
    mean_ar2 = np.mean(agg_r2s)

    results.append({
        "model":         model_name,
        "roc_auc":       round(mean_roc, 4),
        "pr_auc":        round(mean_pr,  4),
        "composite_r2":  round(mean_cr2, 4),
        "agg_r2":        round(mean_ar2, 4),
    })
    print(f"  {model_name:<28} {mean_roc:>8.4f} {mean_pr:>8.4f} "
          f"{mean_cr2:>8.4f} {mean_ar2:>8.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/reports/baseline_comparison.csv", index=False)

print(f"\n{'='*70}")
print("BASELINE COMPARISON SUMMARY")
print("="*70)

best = results_df.sort_values("roc_auc", ascending=False).iloc[0]
xgb_row = results_df[results_df["model"] == "XGBoost (our model)"].iloc[0]
dummy_row = results_df[results_df["model"] == "Majority class (dummy)"].iloc[0]

print(f"\n  Best model:     {best['model']} (ROC-AUC={best['roc_auc']:.4f})")
print(f"  XGBoost:        ROC-AUC={xgb_row['roc_auc']:.4f}  "
      f"Comp R²={xgb_row['composite_r2']:.4f}")
print(f"  Naive baseline: ROC-AUC={dummy_row['roc_auc']:.4f}  "
      f"Comp R²={dummy_row['composite_r2']:.4f}")
print(f"  XGBoost vs naive: "
      f"+{xgb_row['roc_auc']-dummy_row['roc_auc']:.4f} ROC-AUC")

print(f"\n  Full table:")
print(results_df.to_string(index=False))
print(f"\nSaved: outputs/reports/baseline_comparison.csv")
