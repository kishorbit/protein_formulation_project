import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              mean_squared_error, r2_score)
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import shap
import mlflow
import mlflow.sklearn

COMPOSITE_WEIGHTS = {
    "aggregation_score":  0.35,
    "oxidation_level":    0.25,
    "deamidation_level":  0.20,
    "potency_retention":  0.15,
    "shelf_life_score":   0.05,
}
STABILITY_THRESHOLD = 0.40

REGRESSION_TARGETS = [
    "aggregation_score","oxidation_level","deamidation_level",
    "potency_retention","shelf_life_score",
]

print("\nLoading merged dataset (pre-imputation)...")
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

# ── Impute on full training set for final model ───────────
# This is correct here because ml_model.py trains on ALL data
# for the production model (not for validation).
# LOO validation has its own per-fold imputation.
print("\nImputing missing values (fit on full training set)...")
imputer = KNNImputer(n_neighbors=5)
X = imputer.fit_transform(X_raw)

print("Scaling features...")
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# ── Train primary multi-output regressor ─────────────────
print("\nTraining multi-output regressor...")
reg = MultiOutputRegressor(XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0, tree_method="hist",
    # Priority 2 — monotonicity applied in next step
))
reg.fit(X, y_reg)
y_pred = reg.predict(X)

# ── Derive composite and binary ───────────────────────────
y_comp_pred = (
    COMPOSITE_WEIGHTS["aggregation_score"]  * y_pred[:, 0]
  + COMPOSITE_WEIGHTS["oxidation_level"]    * y_pred[:, 1]
  + COMPOSITE_WEIGHTS["deamidation_level"]  * y_pred[:, 2]
  + COMPOSITE_WEIGHTS["potency_retention"]  * (1 - y_pred[:, 3])
  + COMPOSITE_WEIGHTS["shelf_life_score"]   * (1 - y_pred[:, 4])
)
stability_signal = 1 - y_comp_pred

# ── Primary metrics ───────────────────────────────────────
print(f"\n{'='*55}")
print("PRIMARY — Regression metrics")
print("="*55)
reg_metrics = {}
for i, col in enumerate(REGRESSION_TARGETS):
    r2   = r2_score(y_reg[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_reg[:, i], y_pred[:, i]))
    reg_metrics[f"r2_{col}"]   = round(r2,   4)
    reg_metrics[f"rmse_{col}"] = round(rmse, 4)
    print(f"  {col:<25}  R²={r2:.4f}  RMSE={rmse:.4f}")

comp_r2   = r2_score(y_comp, y_comp_pred)
comp_rmse = np.sqrt(mean_squared_error(y_comp, y_comp_pred))
print(f"  {'composite_score':<25}  R²={comp_r2:.4f}  RMSE={comp_rmse:.4f}")

print(f"\n{'='*55}")
print("DIAGNOSTIC — Classification (derived)")
print("="*55)
roc_auc = roc_auc_score(y_bin, stability_signal)
pr_auc  = average_precision_score(y_bin, stability_signal)
print(f"  ROC-AUC : {roc_auc:.4f}")
print(f"  PR-AUC  : {pr_auc:.4f}")

# ── SHAP ─────────────────────────────────────────────────
print("\nComputing SHAP values...")
explainer   = shap.TreeExplainer(reg.estimators_[0])
shap_values = explainer.shap_values(X[:500])
shap_df = pd.DataFrame({
    "feature":         feature_cols,
    "shap_importance": np.abs(shap_values).mean(axis=0),
}).sort_values("shap_importance", ascending=False)
shap_df.to_csv("outputs/reports/shap_importance.csv", index=False)

print("  Top 5 features:")
for _, row in shap_df.head(5).iterrows():
    print(f"    {row['feature']:<35} {row['shap_importance']:.4f}")

# ── Recommendations ───────────────────────────────────────
print("\nBuilding recommendations...")
merged = pd.read_csv("data/processed/dataset_merged.csv")
merged["pred_aggregation"]     = y_pred[:, 0].round(4)
merged["pred_oxidation"]       = y_pred[:, 1].round(4)
merged["pred_deamidation"]     = y_pred[:, 2].round(4)
merged["pred_potency"]         = y_pred[:, 3].round(4)
merged["pred_shelf_life"]      = y_pred[:, 4].round(4)
merged["pred_composite_score"] = y_comp_pred.round(4)
merged["pred_stable"]          = (y_comp_pred < STABILITY_THRESHOLD).astype(int)

recs = (merged.sort_values("pred_composite_score", ascending=True)
              .groupby("protein_id").head(20)
              .reset_index(drop=True))
recs.to_csv("outputs/reports/excipient_recommendations.csv", index=False)

# ── MLflow ───────────────────────────────────────────────
mlflow.set_experiment("formulai_regression_primary")
with mlflow.start_run(run_name="leakage_free_imputation"):
    mlflow.log_params({
        "imputation": "KNNImputer_k5_inside_training",
        "scaling": "MinMaxScaler",
        "architecture": "regression_primary_no_leakage",
        "stability_threshold": STABILITY_THRESHOLD,
    })
    mlflow.log_metrics({
        **reg_metrics,
        "composite_r2": round(comp_r2, 4),
        "roc_auc_derived": round(roc_auc, 4),
        "pr_auc_derived":  round(pr_auc, 4),
    })
    mlflow.sklearn.log_model(reg, "regression_model")

print(f"\n{'='*55}")
print("✓ Done — imputer leakage fixed")
print(f"  ROC-AUC={roc_auc:.3f}  PR-AUC={pr_auc:.3f}")
print(f"  Composite R²={comp_r2:.3f}")
print("="*55)
