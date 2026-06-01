import pandas as pd
import numpy as np
import os, joblib
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, r2_score, mean_absolute_error,
                             mean_squared_error, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve,
                             precision_recall_curve)
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# ── Load & merge ───────────────────────────────────────────
df = pd.read_csv("data/processed/stability_outcomes_v2.csv")
pf = pd.read_csv("data/processed/protein_features_expanded.csv")

df = df.merge(
    pf[["protein_id","isoelectric_point","instability_index","gravy_score",
        "pct_asn","pct_met","pct_cys","pct_trp",
        "met_exposed_fraction","ox_mean_rsa","agg_hotspot_frac",
        "agg_mean","ox_risk_composite","query_label"]],
    on="protein_id"
)

# ── Feature engineering ───────────────────────────────────
CAT_COLS = ["buffer","sugar","surfactant","amino_acid","salt","query_label"]
NUM_COLS = [
    "ph","temperature_c","protein_conc_mgmL",
    "buf_conc_mM","sug_conc_mM","sur_conc_mM","aa_conc_mM","salt_conc_mM",
    "isoelectric_point","instability_index","gravy_score",
    "pct_asn","pct_met","pct_cys","pct_trp",
    "met_exposed_fraction","ox_mean_rsa","agg_hotspot_frac",
    "agg_mean","ox_risk_composite"
]

df["agg_hotspot_frac"] = df["agg_hotspot_frac"].fillna(df["agg_hotspot_frac"].median())

# Interaction features (scientifically motivated)
df["ph_pi_dist"]      = abs(df["ph"] - df["isoelectric_point"])
df["temp_conc"]       = df["temperature_c"] * df["protein_conc_mgmL"]
df["instab_gravy"]    = df["instability_index"] * df["gravy_score"].clip(lower=0)
df["met_temp"]        = df["met_exposed_fraction"] * df["temperature_c"]

INTERACTION_COLS = ["ph_pi_dist","temp_conc","instab_gravy","met_temp"]

df_enc = pd.get_dummies(df[CAT_COLS + NUM_COLS + INTERACTION_COLS],
                        columns=CAT_COLS)
df_enc = df_enc.fillna(df_enc.median())

feature_cols = df_enc.columns.tolist()
X      = df_enc.values.astype(np.float32)
y_cls  = df["stable"].values
y_reg  = df["composite_stability_score"].values
groups = df["protein_id"].values

print(f"Features:  {X.shape[1]}")
print(f"Samples:   {X.shape[0]}")
print(f"Proteins:  {len(np.unique(groups))}")
print(f"Stable:    {y_cls.mean():.3f}\n")

# ── Cross-validation ──────────────────────────────────────
gkf = GroupKFold(n_splits=5)

cls_model = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0, eval_metric="logloss"
)
reg_model = lgb.LGBMRegressor(
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)

# Store OOF predictions
oof_cls_proba = np.zeros(len(X))
oof_cls_pred  = np.zeros(len(X), dtype=int)
oof_reg_pred  = np.zeros(len(X))

cls_metrics, reg_metrics = [], []

print("Running 5-fold GroupKFold CV...")
for fold, (tr, te) in enumerate(gkf.split(X, y_cls, groups)):
    X_tr, X_te   = X[tr], X[te]
    yc_tr, yc_te = y_cls[tr], y_cls[te]
    yr_tr, yr_te = y_reg[tr], y_reg[te]

    # Classification — XGBoost
    cls_model.fit(X_tr, yc_tr)
    proba = cls_model.predict_proba(X_te)[:, 1]
    pred  = cls_model.predict(X_te)
    oof_cls_proba[te] = proba
    oof_cls_pred[te]  = pred

    cls_metrics.append({
        "fold":      fold + 1,
        "AUC-ROC":   roc_auc_score(yc_te, proba),
        "F1":        f1_score(yc_te, pred),
        "Precision": precision_score(yc_te, pred),
        "Recall":    recall_score(yc_te, pred),
    })

    # Regression — LightGBM
    reg_model.fit(X_tr, yr_tr)
    reg_pred = reg_model.predict(X_te)
    oof_reg_pred[te] = reg_pred

    reg_metrics.append({
        "fold":  fold + 1,
        "R2":    r2_score(yr_te, reg_pred),
        "MAE":   mean_absolute_error(yr_te, reg_pred),
        "RMSE":  np.sqrt(mean_squared_error(yr_te, reg_pred)),
    })

    print(f"  Fold {fold+1} | CLS AUC={cls_metrics[-1]['AUC-ROC']:.4f} F1={cls_metrics[-1]['F1']:.4f} "
          f"| REG R2={reg_metrics[-1]['R2']:.4f} MAE={reg_metrics[-1]['MAE']:.4f}")

cls_df = pd.DataFrame(cls_metrics)
reg_df = pd.DataFrame(reg_metrics)

print(f"\n{'='*55}")
print("CLASSIFICATION (XGBoost) — OOF Summary")
print(f"{'='*55}")
print(cls_df.drop("fold",axis=1).agg(["mean","std"]).round(4))

print(f"\n{'='*55}")
print("REGRESSION (LightGBM) — OOF Summary")
print(f"{'='*55}")
print(reg_df.drop("fold",axis=1).agg(["mean","std"]).round(4))

# ── Train final models on full data ───────────────────────
print("\nTraining final models on full dataset...")
cls_final = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0, eval_metric="logloss"
)
reg_final = lgb.LGBMRegressor(
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)
cls_final.fit(X, y_cls)
reg_final.fit(X, y_reg)

joblib.dump(cls_final,    "models/xgb_classifier.pkl")
joblib.dump(reg_final,    "models/lgb_regressor.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")
print("Models saved to models/")

# ── SHAP Analysis ─────────────────────────────────────────
print("\nComputing SHAP values...")

# Classification SHAP (XGBoost — TreeExplainer)
explainer_cls = shap.TreeExplainer(cls_final)
shap_cls      = explainer_cls.shap_values(X)

# Regression SHAP (LightGBM — TreeExplainer)
explainer_reg = shap.TreeExplainer(reg_final)
shap_reg      = explainer_reg.shap_values(X)

# ── Plots ──────────────────────────────────────────────────

# 1. ROC curve
fpr, tpr, _ = roc_curve(y_cls, oof_cls_proba)
auc_score   = roc_auc_score(y_cls, oof_cls_proba)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, lw=2, label=f"XGBoost (AUC = {auc_score:.4f})")
ax.plot([0,1],[0,1],"k--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Stability Classification (OOF)")
ax.legend(); plt.tight_layout()
plt.savefig("outputs/figures/roc_curve.png", dpi=150)
plt.close()

# 2. Precision-Recall curve
prec_arr, rec_arr, _ = precision_recall_curve(y_cls, oof_cls_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec_arr, prec_arr, lw=2)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — Stability Classification (OOF)")
plt.tight_layout()
plt.savefig("outputs/figures/precision_recall_curve.png", dpi=150)
plt.close()

# 3. Confusion matrix
cm = confusion_matrix(y_cls, oof_cls_pred)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["Unstable","Stable"]).plot(ax=ax, colorbar=False)
ax.set_title("Confusion Matrix — XGBoost (OOF)")
plt.tight_layout()
plt.savefig("outputs/figures/confusion_matrix.png", dpi=150)
plt.close()

# 4. Regression: predicted vs actual
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_reg, oof_reg_pred, alpha=0.3, s=8, c="steelblue")
lims = [y_reg.min(), y_reg.max()]
ax.plot(lims, lims, "r--", lw=1)
ax.set_xlabel("Actual Composite Score")
ax.set_ylabel("Predicted Composite Score")
ax.set_title(f"LightGBM Regression — OOF  (R²={r2_score(y_reg,oof_reg_pred):.4f})")
plt.tight_layout()
plt.savefig("outputs/figures/regression_actual_vs_pred.png", dpi=150)
plt.close()

# 5. SHAP summary — Classification (top 20)
fig, ax = plt.subplots(figsize=(9, 7))
shap.summary_plot(shap_cls, X, feature_names=feature_cols,
                  max_display=20, show=False)
plt.title("SHAP Feature Importance — Classification (XGBoost)")
plt.tight_layout()
plt.savefig("outputs/figures/shap_cls_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# 6. SHAP summary — Regression (top 20)
fig, ax = plt.subplots(figsize=(9, 7))
shap.summary_plot(shap_reg, X, feature_names=feature_cols,
                  max_display=20, show=False)
plt.title("SHAP Feature Importance — Regression (LightGBM)")
plt.tight_layout()
plt.savefig("outputs/figures/shap_reg_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# 7. SHAP bar — top 15 mean |SHAP| for both models
for tag, shap_vals, title in [
    ("cls", shap_cls, "Classification — Mean |SHAP|"),
    ("reg", shap_reg, "Regression   — Mean |SHAP|"),
]:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-15:]
    fig, ax  = plt.subplots(figsize=(8, 6))
    ax.barh([feature_cols[i] for i in top_idx],
             mean_abs[top_idx], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top 15 Features — {title}")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/shap_{tag}_bar.png", dpi=150)
    plt.close()

# ── Save OOF predictions ──────────────────────────────────
oof_df = df[["protein_id","ph","temperature_c","protein_conc_mgmL",
             "stable","composite_stability_score"]].copy()
oof_df["pred_stable"]    = oof_cls_pred
oof_df["pred_stable_proba"] = oof_cls_proba.round(4)
oof_df["pred_composite"] = oof_reg_pred.round(4)
oof_df.to_csv("outputs/reports/oof_predictions.csv", index=False)

# ── Save metric tables ────────────────────────────────────
cls_df.to_csv("outputs/reports/cls_cv_metrics.csv", index=False)
reg_df.to_csv("outputs/reports/reg_cv_metrics.csv", index=False)

print("\nAll outputs saved to outputs/")
print("  figures/  — 7 plots")
print("  reports/  — OOF predictions + CV metric tables")
