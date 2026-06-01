import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              mean_squared_error, r2_score)
from xgboost import XGBRegressor
import lightgbm as lgb
import shap
import mlflow
import mlflow.sklearn
import joblib
import os

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
        "met_x_temp":+1,"met_x_ph":+1,"cys_x_ph":+1,"cys_x_temp":+1,
        "cys_met_ratio":+1,"cys_met_sum":+1,
        "flag_high_cys":+1,"flag_high_met":+1,
    },
    "deamidation_level": {
        "temperature_c":+1,"high_temp_flag":+1,"pct_asn":+1,"ph":+1,
        "asn_x_ph":+1,"asn_x_temp":+1,"instab_x_asn":+1,
        "flag_high_asn":+1,
    },
    "potency_retention": {
        "temperature_c":-1,"high_temp_flag":-1,"instability_index":-1,
        "agg_mean":-1,"sug_conc_mM":+1,
        "instab_x_seqlen":-1,"flag_unstable":-1,"flag_long_seq":-1,
    },
    "shelf_life_score": {
        "temperature_c":-1,"high_temp_flag":-1,"instability_index":-1,
        "sug_conc_mM":+1,"sur_conc_mM":+1,
        "instab_x_seqlen":-1,"flag_unstable":-1,"flag_long_seq":-1,
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

print("\nLoading data...")
df = pd.read_csv("data/processed/dataset_merged.csv")
feature_cols = pd.read_csv(
    "data/processed/feature_cols.csv", header=None)[0].tolist()
feature_cols = [c for c in feature_cols if c in df.columns]

print(f"  Dataset shape:   {df.shape}")
print(f"  Feature columns: {len(feature_cols)}")

X_raw  = df[feature_cols].values
y_reg  = df[REGRESSION_TARGETS].values
y_comp = df["composite_stability_score"].values
y_bin  = df["stable"].values

print("\nImputing and scaling...")
imputer = KNNImputer(n_neighbors=5)
X = imputer.fit_transform(X_raw)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

print("\n" + "="*55)
print("LAYER 1 — Training base models")
print("="*55)

print("\n  [1/3] Ridge regression...")
ridge = MultiOutputRegressor(Ridge(alpha=1.0))
ridge.fit(X, y_reg)
y_pred_ridge = ridge.predict(X)
print(f"    Composite R2: {r2_score(y_comp, derive_composite(y_pred_ridge)):.4f}")

print("\n  [2/3] XGBoost with monotonicity constraints...")
xgb_estimators = []
for i, target in enumerate(REGRESSION_TARGETS):
    xgb = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, tree_method="hist",
        monotone_constraints=build_constraint_tuple(target, feature_cols),
    )
    xgb.fit(X, y_reg[:, i])
    xgb_estimators.append(xgb)
y_pred_xgb = np.column_stack([e.predict(X) for e in xgb_estimators])
print(f"    Composite R2: {r2_score(y_comp, derive_composite(y_pred_xgb)):.4f}")

print("\n  [3/3] LightGBM...")
lgb_models = []
for i, target in enumerate(REGRESSION_TARGETS):
    m = lgb.LGBMRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42, verbose=-1)
    m.fit(X, y_reg[:, i])
    lgb_models.append(m)
y_pred_lgb = np.column_stack([m.predict(X) for m in lgb_models])
print(f"    Composite R2: {r2_score(y_comp, derive_composite(y_pred_lgb)):.4f}")

print("\n" + "="*55)
print("LAYER 2 — Training meta-learner")
print("="*55)

meta_features = np.hstack([y_pred_ridge, y_pred_xgb, y_pred_lgb])
meta_feature_names = (
    [f"ridge_{t}"  for t in REGRESSION_TARGETS] +
    [f"xgb_{t}"   for t in REGRESSION_TARGETS] +
    [f"lgb_{t}"   for t in REGRESSION_TARGETS]
)
print(f"\n  Meta-feature matrix shape: {meta_features.shape}")

print("\n  Training meta-learner (Ridge)...")
meta_learner = MultiOutputRegressor(Ridge(alpha=0.5))
meta_learner.fit(meta_features, y_reg)
y_pred_stacked = meta_learner.predict(meta_features)

print("\n" + "="*55)
print("UNCERTAINTY ESTIMATION")
print("="*55)

disagreement = np.mean(np.stack([
    np.abs(y_pred_ridge - y_pred_xgb),
    np.abs(y_pred_xgb   - y_pred_lgb),
    np.abs(y_pred_ridge - y_pred_lgb),
], axis=0), axis=(0, 2))

disagreement_norm = (disagreement - disagreement.min()) / \
                    (disagreement.max() - disagreement.min() + 1e-8)

confidence = np.where(disagreement_norm < 0.33, "High",
             np.where(disagreement_norm < 0.66, "Medium", "Low"))

print(f"  High confidence:   {(confidence=='High').sum()} samples")
print(f"  Medium confidence: {(confidence=='Medium').sum()} samples")
print(f"  Low confidence:    {(confidence=='Low').sum()} samples")

y_comp_pred      = derive_composite(y_pred_stacked)
stability_signal = 1 - y_comp_pred

print(f"\n{'='*55}")
print("STACKED ENSEMBLE — Final metrics")
print("="*55)

reg_metrics = {}
for i, col in enumerate(REGRESSION_TARGETS):
    r2   = r2_score(y_reg[:, i], y_pred_stacked[:, i])
    rmse = np.sqrt(mean_squared_error(y_reg[:, i], y_pred_stacked[:, i]))
    reg_metrics[f"r2_{col}"]   = round(r2,   4)
    reg_metrics[f"rmse_{col}"] = round(rmse, 4)
    print(f"  {col:<25}  R2={r2:.4f}  RMSE={rmse:.4f}")

comp_r2   = r2_score(y_comp, y_comp_pred)
comp_rmse = np.sqrt(mean_squared_error(y_comp, y_comp_pred))
print(f"  {'composite_score':<25}  R2={comp_r2:.4f}  RMSE={comp_rmse:.4f}")

roc_auc = roc_auc_score(y_bin, stability_signal)
pr_auc  = average_precision_score(y_bin, stability_signal)
print(f"\n  ROC-AUC : {roc_auc:.4f}")
print(f"  PR-AUC  : {pr_auc:.4f}")

print(f"\n{'='*55}")
print("MODEL COMPARISON")
print("="*55)
for name, y_p in [("Ridge",    y_pred_ridge),
                   ("XGBoost",  y_pred_xgb),
                   ("LightGBM", y_pred_lgb),
                   ("Stacked",  y_pred_stacked)]:
    cp = derive_composite(y_p)
    print(f"  {name:<12}  Comp R2={r2_score(y_comp,cp):.4f}"
          f"  Agg R2={r2_score(y_reg[:,0],y_p[:,0]):.4f}"
          f"  ROC={roc_auc_score(y_bin,1-cp):.4f}")

print("\nBuilding recommendations with uncertainty scores...")
merged = pd.read_csv("data/processed/dataset_merged.csv")
merged["pred_aggregation"]     = y_pred_stacked[:, 0].round(4)
merged["pred_oxidation"]       = y_pred_stacked[:, 1].round(4)
merged["pred_deamidation"]     = y_pred_stacked[:, 2].round(4)
merged["pred_potency"]         = y_pred_stacked[:, 3].round(4)
merged["pred_shelf_life"]      = y_pred_stacked[:, 4].round(4)
merged["pred_composite_score"] = y_comp_pred.round(4)
merged["pred_stable"]          = (y_comp_pred < STABILITY_THRESHOLD).astype(int)
merged["uncertainty"]          = disagreement_norm.round(4)
merged["confidence"]           = confidence

recs = (merged.sort_values("pred_composite_score", ascending=True)
              .groupby("protein_id").head(20)
              .reset_index(drop=True))
recs.to_csv("outputs/reports/excipient_recommendations.csv", index=False)

os.makedirs("outputs/models", exist_ok=True)
joblib.dump(imputer,        "outputs/models/imputer.pkl")
joblib.dump(scaler,         "outputs/models/scaler.pkl")
joblib.dump(ridge,          "outputs/models/ridge_base.pkl")
joblib.dump(xgb_estimators, "outputs/models/xgb_base.pkl")
joblib.dump(lgb_models,     "outputs/models/lgb_base.pkl")
joblib.dump(meta_learner,   "outputs/models/meta_learner.pkl")
print("  Models saved to outputs/models/")

mlflow.set_experiment("formulai_stacked_ensemble")
with mlflow.start_run(run_name="stacking_ridge_xgb_lgb"):
    mlflow.log_params({
        "architecture":    "stacking_layer2_ridge",
        "base_models":     "ridge+xgboost+lightgbm",
        "meta_learner":    "ridge_alpha_0.5",
        "n_meta_features": 15,
    })
    mlflow.log_metrics({
        **reg_metrics,
        "composite_r2":    round(comp_r2,  4),
        "roc_auc_derived": round(roc_auc,  4),
        "pr_auc_derived":  round(pr_auc,   4),
    })

print(f"\n{'='*55}")
print("Done — Stacked ensemble complete")
print(f"  Composite R2={comp_r2:.4f}  ROC-AUC={roc_auc:.4f}")
print(f"  Uncertainty scores added to recommendations")
print(f"  Models saved: outputs/models/")
print("="*55)
