import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import lightgbm as lgb

TARGETS = ["aggregation_score","oxidation_level","deamidation_level",
           "potency_retention","shelf_life_score"]
PRED    = ["pred_aggregation","pred_oxidation","pred_deamidation",
           "pred_potency","pred_shelf_life"]
COMPOSITE_WEIGHTS = {
    "aggregation_score":0.35,"oxidation_level":0.25,
    "deamidation_level":0.20,"potency_retention":0.15,"shelf_life_score":0.05,
}
NAMED_CONSTRAINTS = {
    "aggregation_score": {"temperature_c":+1,"high_temp_flag":+1,"instability_index":+1,
        "agg_mean":+1,"agg_hotspot_frac":+1,"sug_conc_mM":-1,"sur_conc_mM":-1,"sug_x_instability":-1},
    "oxidation_level": {"temperature_c":+1,"high_temp_flag":+1,"pct_met":+1,"pct_trp":+1},
    "deamidation_level": {"temperature_c":+1,"high_temp_flag":+1,"pct_asn":+1,"ph":+1},
    "potency_retention": {"temperature_c":-1,"high_temp_flag":-1,"instability_index":-1,
        "agg_mean":-1,"sug_conc_mM":+1},
    "shelf_life_score": {"temperature_c":-1,"high_temp_flag":-1,"instability_index":-1,
        "sug_conc_mM":+1,"sur_conc_mM":+1},
}

def build_constraint_tuple(target, feature_cols):
    c = NAMED_CONSTRAINTS.get(target, {})
    return tuple(c.get(f, 0) for f in feature_cols)

def derive_composite(y_pred):
    return (COMPOSITE_WEIGHTS["aggregation_score"]  * y_pred[:,0]
          + COMPOSITE_WEIGHTS["oxidation_level"]    * y_pred[:,1]
          + COMPOSITE_WEIGHTS["deamidation_level"]  * y_pred[:,2]
          + COMPOSITE_WEIGHTS["potency_retention"]  * (1-y_pred[:,3])
          + COMPOSITE_WEIGHTS["shelf_life_score"]   * (1-y_pred[:,4]))

print("\nLoading data...")
df = pd.read_csv("data/processed/dataset_merged.csv")
feature_cols = pd.read_csv(
    "data/processed/feature_cols.csv", header=None)[0].tolist()
feature_cols = [c for c in feature_cols if c in df.columns]
proteins = df["protein_id"].unique()

print(f"  Proteins: {len(proteins)}  Features: {len(feature_cols)}")

# ── LOO loop to collect residuals ─────────────────────────
print("\nRunning LOO to collect per-fold residuals for bias estimation...")
all_true = []
all_pred = []

for held_out in proteins:
    train_mask = df["protein_id"] != held_out
    test_mask  = df["protein_id"] == held_out

    X_tr_raw = df.loc[train_mask, feature_cols].values
    X_te_raw = df.loc[test_mask,  feature_cols].values
    y_tr     = df.loc[train_mask, TARGETS].values
    y_te     = df.loc[test_mask,  TARGETS].values

    imp = KNNImputer(n_neighbors=5)
    X_tr = imp.fit_transform(X_tr_raw)
    X_te = imp.transform(X_te_raw)
    scl = MinMaxScaler()
    X_tr = scl.fit_transform(X_tr)
    X_te = scl.transform(X_te)

    ridge = MultiOutputRegressor(Ridge(alpha=1.0))
    ridge.fit(X_tr, y_tr)

    xgb_ests = []
    for i, target in enumerate(TARGETS):
        xgb = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, tree_method="hist",
            monotone_constraints=build_constraint_tuple(target, feature_cols))
        xgb.fit(X_tr, y_tr[:,i])
        xgb_ests.append(xgb)

    lgb_ests = []
    for i, target in enumerate(TARGETS):
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=42, verbose=-1)
        m.fit(X_tr, y_tr[:,i])
        lgb_ests.append(m)

    meta_tr = np.hstack([ridge.predict(X_tr),
        np.column_stack([e.predict(X_tr) for e in xgb_ests]),
        np.column_stack([e.predict(X_tr) for e in lgb_ests])])
    meta_learner = MultiOutputRegressor(Ridge(alpha=0.5))
    meta_learner.fit(meta_tr, y_tr)

    meta_te = np.hstack([ridge.predict(X_te),
        np.column_stack([e.predict(X_te) for e in xgb_ests]),
        np.column_stack([e.predict(X_te) for e in lgb_ests])])
    y_pred = meta_learner.predict(meta_te)

    all_true.append(y_te)
    all_pred.append(y_pred)

all_true = np.vstack(all_true)
all_pred = np.vstack(all_pred)

# ── Compute per-target bias ───────────────────────────────
print(f"\n{'='*55}")
print("BIAS ESTIMATION (LOO residuals)")
print("="*55)
biases = {}
for i, target in enumerate(TARGETS):
    bias = (all_pred[:,i] - all_true[:,i]).mean()
    biases[target] = bias
    print(f"  {target:<25} bias={bias:+.4f}")

# ── Save bias corrections ─────────────────────────────────
bias_df = pd.DataFrame([{
    "target": t, "bias": round(b, 6)
} for t,b in biases.items()])
bias_df.to_csv("outputs/reports/bias_corrections.csv", index=False)

# ── Apply correction to existing recommendations ──────────
print("\nApplying bias correction to recommendations...")
recs = pd.read_csv("outputs/reports/excipient_recommendations.csv")
pred_map = dict(zip(TARGETS, PRED))

for target, bias in biases.items():
    pred_col = pred_map[target]
    if pred_col in recs.columns:
        recs[pred_col] = (recs[pred_col] - bias).clip(0, 1)

# Recompute composite
y_corrected = recs[PRED].values
comp = derive_composite(y_corrected)
recs["pred_composite_score"] = comp.round(4)
recs.to_csv("outputs/reports/excipient_recommendations.csv", index=False)

# ── Apply to pareto ───────────────────────────────────────
pareto = pd.read_csv("outputs/reports/pareto_recommendations.csv")
for target, bias in biases.items():
    pred_col = pred_map[target]
    if pred_col in pareto.columns:
        pareto[pred_col] = (pareto[pred_col] - bias).clip(0, 1)
if all(c in pareto.columns for c in PRED):
    pareto["composite_stability_score"] = derive_composite(
        pareto[PRED].values).round(4)
pareto.to_csv("outputs/reports/pareto_recommendations.csv", index=False)

# ── Verify correction ─────────────────────────────────────
print(f"\n  Verification — residuals after correction:")
for i, target in enumerate(TARGETS):
    corrected_pred = all_pred[:,i] - biases[target]
    new_bias = (corrected_pred - all_true[:,i]).mean()
    r2 = r2_score(all_true[:,i], corrected_pred)
    print(f"  {target:<25} bias={new_bias:+.6f}  R²={r2:.4f}")

print(f"\nSaved: outputs/reports/bias_corrections.csv")
print(f"Updated: outputs/reports/excipient_recommendations.csv")
print(f"Updated: outputs/reports/pareto_recommendations.csv")
print("="*55)
