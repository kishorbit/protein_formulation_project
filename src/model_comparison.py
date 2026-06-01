import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import (make_scorer, roc_auc_score, f1_score,
                             r2_score, mean_absolute_error, mean_squared_error)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ── Load ──────────────────────────────────────────────────
df = pd.read_csv("data/processed/stability_outcomes_v2.csv")
pf = pd.read_csv("data/processed/protein_features_expanded.csv")

df = df.merge(
    pf[["protein_id","isoelectric_point","instability_index","gravy_score",
        "pct_asn","pct_met","pct_cys","pct_trp",
        "met_exposed_fraction","ox_mean_rsa","agg_hotspot_frac",
        "agg_mean","ox_risk_composite","query_label"]],
    on="protein_id", suffixes=("","_pf")
)

# ── Features ──────────────────────────────────────────────
CAT_COLS = ["buffer","sugar","surfactant","amino_acid","salt","query_label"]
NUM_COLS = [
    "ph","temperature_c","protein_conc_mgmL",
    "buf_conc_mM","sug_conc_mM","sur_conc_mM","aa_conc_mM","salt_conc_mM",
    "isoelectric_point","instability_index","gravy_score",
    "pct_asn","pct_met","pct_cys","pct_trp",
    "met_exposed_fraction","ox_mean_rsa","agg_hotspot_frac",
    "agg_mean","ox_risk_composite"
]

# Fill NaN in agg_hotspot_frac (27/79 proteins)
df["agg_hotspot_frac"] = df["agg_hotspot_frac"].fillna(df["agg_hotspot_frac"].median())

# One-hot encode categoricals
df_enc = pd.get_dummies(df[CAT_COLS + NUM_COLS], columns=CAT_COLS)
feature_cols = df_enc.columns.tolist()

# Fill any remaining NaNs (median for numerics, 0 for one-hot)
df_enc = df_enc.fillna(df_enc.median())
X = df_enc.values.astype(np.float32)
print(f'NaNs in X after fill: {np.isnan(X).sum()}')
y_cls = df["stable"].values
y_reg = df["composite_stability_score"].values
groups = df["protein_id"].values

print(f"Features: {X.shape[1]}")
print(f"Samples:  {X.shape[0]}")
print(f"Proteins: {len(np.unique(groups))}\n")

# ── CV strategy ───────────────────────────────────────────
gkf = GroupKFold(n_splits=5)

# ── Model definitions ─────────────────────────────────────
cls_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0),
    "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "ExtraTrees":         ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost":            xgb.XGBClassifier(n_estimators=300, learning_rate=0.05,
                              max_depth=6, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbosity=0, eval_metric="logloss"),
    "LightGBM":           lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                              num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbose=-1),
}

reg_models = {
    "Ridge":           Ridge(alpha=1.0),
    "RandomForest":    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "ExtraTrees":      ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost":         xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                           max_depth=6, subsample=0.8, colsample_bytree=0.8,
                           random_state=42, verbosity=0),
    "LightGBM":        lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                           num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                           random_state=42, verbose=-1),
}

# ── Classification CV ─────────────────────────────────────
print("=" * 60)
print("CLASSIFICATION (predicting 'stable')")
print("=" * 60)

cls_results = []
for name, model in cls_models.items():
    scores = cross_validate(
        model, X, y_cls, groups=groups, cv=gkf,
        scoring={
            "auc":       make_scorer(roc_auc_score, needs_proba=True),
            "f1":        make_scorer(f1_score),
            "precision": "precision",
            "recall":    "recall",
        },
        n_jobs=-1 if name not in ["XGBoost","LightGBM"] else 1
    )
    cls_results.append({
        "Model":     name,
        "AUC-ROC":   scores["test_auc"].mean(),
        "AUC-std":   scores["test_auc"].std(),
        "F1":        scores["test_f1"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall":    scores["test_recall"].mean(),
    })
    print(f"  {name:<22} AUC={scores['test_auc'].mean():.4f}±{scores['test_auc'].std():.4f}  F1={scores['test_f1'].mean():.4f}")

cls_df = pd.DataFrame(cls_results).sort_values("AUC-ROC", ascending=False)

# ── Regression CV ─────────────────────────────────────────
print()
print("=" * 60)
print("REGRESSION (predicting 'composite_stability_score')")
print("=" * 60)

reg_results = []
for name, model in reg_models.items():
    scores = cross_validate(
        model, X, y_reg, groups=groups, cv=gkf,
        scoring={
            "r2":  "r2",
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "mse": make_scorer(mean_squared_error, greater_is_better=False),
        },
        n_jobs=-1 if name not in ["XGBoost","LightGBM"] else 1
    )
    rmse = np.sqrt(-scores["test_mse"].mean())
    reg_results.append({
        "Model": name,
        "R2":    scores["test_r2"].mean(),
        "R2-std": scores["test_r2"].std(),
        "MAE":   -scores["test_mae"].mean(),
        "RMSE":  rmse,
    })
    print(f"  {name:<22} R2={scores['test_r2'].mean():.4f}±{scores['test_r2'].std():.4f}  MAE={-scores['test_mae'].mean():.4f}  RMSE={rmse:.4f}")

reg_df = pd.DataFrame(reg_results).sort_values("R2", ascending=False)

# ── Summary tables ────────────────────────────────────────
print()
print("=" * 60)
print("CLASSIFICATION SUMMARY (sorted by AUC-ROC)")
print("=" * 60)
print(cls_df.round(4).to_string(index=False))

print()
print("=" * 60)
print("REGRESSION SUMMARY (sorted by R2)")
print("=" * 60)
print(reg_df.round(4).to_string(index=False))

# Save
cls_df.to_csv("data/processed/model_comparison_cls.csv", index=False)
reg_df.to_csv("data/processed/model_comparison_reg.csv", index=False)
print("\nSaved comparison tables.")
