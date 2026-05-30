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
    "aggregation_score": {"temperature_c":+1,"high_temp_flag":+1,"instability_index":+1,"agg_mean":+1,"agg_hotspot_frac":+1,"sug_conc_mM":-1,"sur_conc_mM":-1,"sug_x_instability":-1},
    "oxidation_level": {"temperature_c":+1,"high_temp_flag":+1,"pct_met":+1,"pct_trp":+1},
    "deamidation_level": {"temperature_c":+1,"high_temp_flag":+1,"pct_asn":+1,"ph":+1},
    "potency_retention": {"temperature_c":-1,"high_temp_flag":-1,"instability_index":-1,"agg_mean":-1,"sug_conc_mM":+1},
    "shelf_life_score": {"temperature_c":-1,"high_temp_flag":-1,"instability_index":-1,"sug_conc_mM":+1,"sur_conc_mM":+1},
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

proteins = df["protein_id"].unique()
print(f"  Proteins: {len(proteins)}  |  Features: {len(feature_cols)}")

results = []
print("\nRunning LOO validation — stacked ensemble...")
print(f"{'Protein':<16} {'ROC':>7} {'PR':>7} {'CompR2':>8} {'AggR2':>7} {'Conf'}")
print("-" * 60)

for held_out in proteins:
    train_mask = df["protein_id"] != held_out
    test_mask  = df["protein_id"] == held_out

    X_tr_raw    = df.loc[train_mask, feature_cols].values
    X_te_raw    = df.loc[test_mask,  feature_cols].values
    y_tr        = df.loc[train_mask, REGRESSION_TARGETS].values
    y_te        = df.loc[test_mask,  REGRESSION_TARGETS].values
    y_bin       = df.loc[test_mask,  "stable"].values
    y_comp_true = df.loc[test_mask,  "composite_stability_score"].values

    imp = KNNImputer(n_neighbors=5)
    X_tr = imp.fit_transform(X_tr_raw)
    X_te = imp.transform(X_te_raw)
    scl = MinMaxScaler()
    X_tr = scl.fit_transform(X_tr)
    X_te = scl.transform(X_te)

    # Layer 1
    ridge = MultiOutputRegressor(Ridge(alpha=1.0))
    ridge.fit(X_tr, y_tr)

    xgb_ests = []
    for i, target in enumerate(REGRESSION_TARGETS):
        xgb = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, tree_method="hist",
            monotone_constraints=build_constraint_tuple(target, feature_cols))
        xgb.fit(X_tr, y_tr[:, i])
        xgb_ests.append(xgb)

    lgb_ests = []
    for i, target in enumerate(REGRESSION_TARGETS):
        m = lgb.LGBMRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42, verbose=-1)
        m.fit(X_tr, y_tr[:, i])
        lgb_ests.append(m)

    # Meta-learner trained on train fold predictions
    meta_tr = np.hstack([
        ridge.predict(X_tr),
        np.column_stack([e.predict(X_tr) for e in xgb_ests]),
        np.column_stack([e.predict(X_tr) for e in lgb_ests]),
    ])
    meta_learner = MultiOutputRegressor(Ridge(alpha=0.5))
    meta_learner.fit(meta_tr, y_tr)

    # Test fold predictions
    meta_te = np.hstack([
        ridge.predict(X_te),
        np.column_stack([e.predict(X_te) for e in xgb_ests]),
        np.column_stack([e.predict(X_te) for e in lgb_ests]),
    ])
    y_pred = meta_learner.predict(meta_te)

    # Uncertainty
    y_r = ridge.predict(X_te)
    y_x = np.column_stack([e.predict(X_te) for e in xgb_ests])
    y_l = np.column_stack([e.predict(X_te) for e in lgb_ests])
    disagreement = np.mean(np.stack([
        np.abs(y_r-y_x),np.abs(y_x-y_l),np.abs(y_r-y_l)
    ], axis=0), axis=(0,2))
    mean_unc = disagreement.mean()
    conf = "High" if mean_unc < 0.02 else "Medium" if mean_unc < 0.04 else "Low"

    comp_pred = derive_composite(y_pred)
    signal    = 1 - comp_pred
    n_cls     = len(np.unique(y_bin))

    roc_auc = roc_auc_score(y_bin, signal) if n_cls > 1 else float("nan")
    pr_auc  = average_precision_score(y_bin, signal) if n_cls > 1 else float("nan")
    comp_r2 = r2_score(y_comp_true, comp_pred)
    agg_r2  = r2_score(y_te[:, 0], y_pred[:, 0])

    reg_metrics = {}
    for i, col in enumerate(REGRESSION_TARGETS):
        reg_metrics[f"r2_{col}"]   = round(r2_score(y_te[:,i], y_pred[:,i]), 3)
        reg_metrics[f"rmse_{col}"] = round(np.sqrt(mean_squared_error(
            y_te[:,i], y_pred[:,i])), 3)

    results.append({
        "held_out_protein": held_out,
        "test_n":           int(test_mask.sum()),
        "roc_auc":          round(roc_auc, 3),
        "pr_auc":           round(pr_auc,  3),
        "composite_r2":     round(comp_r2, 3),
        "agg_r2":           round(agg_r2,  3),
        "mean_uncertainty": round(mean_unc, 4),
        "confidence":       conf,
        **reg_metrics,
    })
    print(f"  {held_out:<14} {roc_auc:>7.3f} {pr_auc:>7.3f} "
          f"{comp_r2:>8.3f} {agg_r2:>7.3f}  {conf}")

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/reports/held_out_validation.csv", index=False)

print(f"\n{'='*55}")
print("HELD-OUT VALIDATION — stacked ensemble")
print("="*55)
print(f"  Mean ROC-AUC:      {results_df['roc_auc'].dropna().mean():.4f}")
print(f"  Mean PR-AUC:       {results_df['pr_auc'].dropna().mean():.4f}")
print(f"  Mean Composite R2: {results_df['composite_r2'].mean():.4f}")
print(f"  Mean Agg R2:       {results_df['agg_r2'].mean():.4f}")
print(f"  Min ROC-AUC:       {results_df['roc_auc'].min():.4f}")
print(f"  High confidence:   {(results_df['confidence']=='High').sum()}/{len(results_df)} proteins")
print(f"\nSaved: outputs/reports/held_out_validation.csv")
