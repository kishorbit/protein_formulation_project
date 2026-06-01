import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

TARGETS = [
    "aggregation_score","oxidation_level","deamidation_level",
    "potency_retention","shelf_life_score",
]
DEFAULT_WEIGHTS = [0.35, 0.25, 0.20, 0.15, 0.05]
N_ITER = 500

print("\nLoading predictions...")
recs = pd.read_csv("outputs/reports/excipient_recommendations.csv")

pred_cols = ["pred_aggregation","pred_oxidation","pred_deamidation",
             "pred_potency","pred_shelf_life"]
true_cols = ["aggregation_score","oxidation_level","deamidation_level",
             "potency_retention","shelf_life_score"]

for c in pred_cols + true_cols + ["composite_stability_score"]:
    if c not in recs.columns:
        print(f"  Missing column: {c}")
        raise SystemExit(1)

Y_pred     = recs[pred_cols].values
Y_true     = recs[true_cols].values
Y_true_comp = recs["composite_stability_score"].values

print(f"  Rows aligned: {len(recs)}")

def composite(Y, w):
    return (w[0]*Y[:,0] + w[1]*Y[:,1] + w[2]*Y[:,2]
          + w[3]*(1-Y[:,3]) + w[4]*(1-Y[:,4]))

default_comp = composite(Y_pred, DEFAULT_WEIGHTS)
default_r2   = r2_score(Y_true_comp, default_comp)
print(f"  Default weights composite R²: {default_r2:.4f}")

print(f"\nRunning {N_ITER} Monte Carlo weight perturbations...")
results = []
for _ in range(N_ITER):
    w    = np.random.dirichlet(np.ones(5))
    comp = composite(Y_pred, w)
    r2   = r2_score(Y_true_comp, comp)
    results.append({
        "w_aggregation":  round(w[0], 4),
        "w_oxidation":    round(w[1], 4),
        "w_deamidation":  round(w[2], 4),
        "w_potency":      round(w[3], 4),
        "w_shelf_life":   round(w[4], 4),
        "composite_r2":   round(r2,   4),
        "composite_mean": round(comp.mean(), 4),
        "composite_std":  round(comp.std(),  4),
    })

res_df = pd.DataFrame(results)
res_df.to_csv("outputs/reports/weight_sensitivity.csv", index=False)

print(f"\n{'='*55}")
print("WEIGHT SENSITIVITY ANALYSIS")
print("="*55)
print(f"  Default composite R²:   {default_r2:.4f}")
print(f"  MC mean composite R²:   {res_df['composite_r2'].mean():.4f}")
print(f"  MC std  composite R²:   {res_df['composite_r2'].std():.4f}")
print(f"  MC min  composite R²:   {res_df['composite_r2'].min():.4f}")
print(f"  MC max  composite R²:   {res_df['composite_r2'].max():.4f}")
print(f"\n  Composite score range across all weight configs:")
print(f"    Mean score min: {res_df['composite_mean'].min():.4f}")
print(f"    Mean score max: {res_df['composite_mean'].max():.4f}")
print(f"    Spread:         {res_df['composite_mean'].max()-res_df['composite_mean'].min():.4f}")

spread = res_df['composite_mean'].max() - res_df['composite_mean'].min()
print(f"\n  Interpretation:")
if spread < 0.05:
    print("    LOW sensitivity — weight choice has minimal effect on rankings")
elif spread < 0.15:
    print("    MODERATE sensitivity — weights matter, user control is important")
else:
    print("    HIGH sensitivity — weight choice significantly changes recommendations")

print(f"\n  Per-target prediction quality (vs true values):")
for i, (pc, tc) in enumerate(zip(pred_cols, true_cols)):
    r2 = r2_score(Y_true[:,i], Y_pred[:,i])
    print(f"    {tc:<25} R²={r2:.4f}")

print(f"\nSaved: outputs/reports/weight_sensitivity.csv")
