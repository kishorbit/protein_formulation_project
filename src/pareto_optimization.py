import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
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

def is_pareto_efficient(costs):
    n = len(costs)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]: continue
        dominated = (np.all(costs >= costs[i], axis=1) &
                     np.any(costs >  costs[i], axis=1))
        dominated[i] = False
        is_efficient[dominated] = False
    return is_efficient

print("\nLoading data...")
df = pd.read_csv("data/processed/dataset_merged.csv")
feature_cols = pd.read_csv(
    "data/processed/feature_cols.csv", header=None)[0].tolist()
feature_cols = [c for c in feature_cols if c in df.columns]

print(f"  Dataset shape:   {df.shape}")
print(f"  Feature columns: {len(feature_cols)}")

X_raw = df[feature_cols].values
y_reg = df[REGRESSION_TARGETS].values

print("\nImputing and scaling...")
imputer = KNNImputer(n_neighbors=5)
X = imputer.fit_transform(X_raw)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

print("\nTraining stacked ensemble...")

print("  [1/3] Ridge...")
ridge = MultiOutputRegressor(Ridge(alpha=1.0))
ridge.fit(X, y_reg)
y_pred_ridge = ridge.predict(X)

print("  [2/3] XGBoost...")
xgb_ests = []
for i, target in enumerate(REGRESSION_TARGETS):
    xgb = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, tree_method="hist",
        monotone_constraints=build_constraint_tuple(target, feature_cols))
    xgb.fit(X, y_reg[:, i])
    xgb_ests.append(xgb)
y_pred_xgb = np.column_stack([e.predict(X) for e in xgb_ests])

print("  [3/3] LightGBM...")
lgb_ests = []
for i, target in enumerate(REGRESSION_TARGETS):
    m = lgb.LGBMRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42, verbose=-1)
    m.fit(X, y_reg[:, i])
    lgb_ests.append(m)
y_pred_lgb = np.column_stack([m.predict(X) for m in lgb_ests])

print("  [Meta] Training meta-learner...")
meta_features = np.hstack([y_pred_ridge, y_pred_xgb, y_pred_lgb])
meta_learner  = MultiOutputRegressor(Ridge(alpha=0.5))
meta_learner.fit(meta_features, y_reg)
y_pred_stacked = meta_learner.predict(meta_features)

# Uncertainty
disagreement = np.mean(np.stack([
    np.abs(y_pred_ridge - y_pred_xgb),
    np.abs(y_pred_xgb   - y_pred_lgb),
    np.abs(y_pred_ridge - y_pred_lgb),
], axis=0), axis=(0, 2))
disagreement_norm = (disagreement - disagreement.min()) / \
                    (disagreement.max() - disagreement.min() + 1e-8)
confidence = np.where(disagreement_norm < 0.33, "High",
             np.where(disagreement_norm < 0.66, "Medium", "Low"))

y_comp_pred = derive_composite(y_pred_stacked)

# Pareto objectives — all higher = better
obj = pd.DataFrame({
    "stability_score":  1 - y_comp_pred,
    "low_aggregation":  1 - y_pred_stacked[:, 0],
    "low_oxidation":    1 - y_pred_stacked[:, 1],
    "low_deamidation":  1 - y_pred_stacked[:, 2],
    "high_potency":     y_pred_stacked[:, 3],
    "high_shelf_life":  y_pred_stacked[:, 4],
})

print("\nComputing Pareto front...")
pareto_mask = is_pareto_efficient(obj.values)
print(f"  Pareto-efficient: {pareto_mask.sum()} / {len(obj)}")

merged = pd.read_csv("data/processed/dataset_merged.csv")
pareto_df = merged[pareto_mask].copy().reset_index(drop=True)

for col in obj.columns:
    pareto_df[col] = obj[col].values[pareto_mask]

pareto_df["composite_stability_score"] = y_comp_pred[pareto_mask].round(4)
pareto_df["uncertainty"]               = disagreement_norm[pareto_mask].round(4)
pareto_df["confidence"]                = confidence[pareto_mask]
pareto_df["pred_aggregation"]          = y_pred_stacked[pareto_mask, 0].round(4)
pareto_df["pred_oxidation"]            = y_pred_stacked[pareto_mask, 1].round(4)
pareto_df["pred_deamidation"]          = y_pred_stacked[pareto_mask, 2].round(4)
pareto_df["pred_potency"]              = y_pred_stacked[pareto_mask, 3].round(4)
pareto_df["pred_shelf_life"]           = y_pred_stacked[pareto_mask, 4].round(4)

pareto_df = pareto_df.sort_values("composite_stability_score", ascending=True)

keep = ["protein_id","buffer","sugar","surfactant","amino_acid","ph",
        "temperature_c","composite_stability_score","stability_score",
        "low_aggregation","low_oxidation","low_deamidation",
        "high_potency","high_shelf_life",
        "pred_aggregation","pred_oxidation","pred_deamidation",
        "pred_potency","pred_shelf_life",
        "uncertainty","confidence"]
keep = [c for c in keep if c in pareto_df.columns]
pareto_out = pareto_df[keep].reset_index(drop=True)
pareto_out.to_csv("outputs/reports/pareto_recommendations.csv", index=False)

print(f"\n{'='*65}")
print("PARETO RECOMMENDATIONS — stacked ensemble")
print("="*65)
for pid in pareto_out["protein_id"].unique()[:4]:
    sub = pareto_out[pareto_out["protein_id"]==pid].head(3)
    print(f"\n  {pid}")
    print(f"  {'Buffer':<16} {'Sugar':<12} {'Composite':>10} {'Confidence':>12}")
    print(f"  {'-'*54}")
    for _, r in sub.iterrows():
        print(f"  {r['buffer']:<16} {r['sugar']:<12} "
              f"{r['composite_stability_score']:>10.4f} "
              f"{r['confidence']:>12}")

print(f"\nSaved: outputs/reports/pareto_recommendations.csv")
