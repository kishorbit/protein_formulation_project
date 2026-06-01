import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os

os.makedirs("outputs/reports", exist_ok=True)

print("\nLoading data...")
val_df  = pd.read_csv("outputs/reports/stacked_validation.csv")
prot    = pd.read_csv("data/processed/protein_features.csv")
bias_df = pd.read_csv("outputs/reports/bias_corrections.csv")

print(f"  Validation rows: {len(val_df)}")

plt.rcParams.update({
    "figure.facecolor":"#05070d","axes.facecolor":"#0a0e1a",
    "axes.edgecolor":"#1e2535","axes.labelcolor":"#6b748a",
    "xtick.color":"#6b748a","ytick.color":"#6b748a",
    "text.color":"#e2e8f0","grid.color":"#1e2535",
    "grid.linewidth":0.5,"font.size":9,
})
CYAN="#00c8f0"; GREEN="#00d68a"; AMBER="#f0a020"
VIOLET="#7c6af7"; ROSE="#f05070"
COLORS = [ROSE, AMBER, VIOLET, GREEN, CYAN]

TARGETS = ["aggregation_score","oxidation_level","deamidation_level",
           "potency_retention","shelf_life_score"]
R2_COLS = ["r2_aggregation_score","r2_oxidation_level","r2_deamidation_level",
           "r2_potency_retention","r2_shelf_life_score"]
RMSE_COLS = ["rmse_aggregation_score","rmse_oxidation_level","rmse_deamidation_level",
             "rmse_potency_retention","rmse_shelf_life_score"]

# ── Use per-protein LOO metrics as residual proxy ─────────
# R² and RMSE per protein per target is already in stacked_validation.csv
# This is the only correctly aligned source

print("\n  Per-target LOO metrics (correctly aligned):")
print(f"  {'Target':<25} {'Mean R²':>9} {'Std R²':>8} {'Mean RMSE':>10}")
print(f"  {'-'*55}")

target_stats = []
for target, r2_col, rmse_col in zip(TARGETS, R2_COLS, RMSE_COLS):
    if r2_col in val_df.columns:
        r2s   = val_df[r2_col].values
        rmses = val_df[rmse_col].values if rmse_col in val_df.columns else np.full(len(val_df), np.nan)
        target_stats.append({
            "target":    target,
            "mean_r2":   r2s.mean(),
            "std_r2":    r2s.std(),
            "min_r2":    r2s.min(),
            "mean_rmse": rmses.mean(),
        })
        print(f"  {target:<25} {r2s.mean():>9.4f} {r2s.std():>8.4f} {rmses.mean():>10.4f}")

# ── Plot 1 — Per-protein R² per target ───────────────────
print("\nPlotting per-protein R² distributions...")
fig, axes = plt.subplots(1, len(TARGETS), figsize=(16, 3.5))
for ax, target, r2_col, color in zip(axes, TARGETS, R2_COLS, COLORS):
    if r2_col in val_df.columns:
        vals = val_df[r2_col].values
        ax.hist(vals, bins=15, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(vals.mean(), color="white", lw=1.5, ls="--",
                   label=f"mean={vals.mean():.3f}")
        ax.axvline(0, color=ROSE, lw=1, ls=":", alpha=0.6)
    ax.set_title(target.replace("_"," "), fontsize=8, pad=6)
    ax.set_xlabel("LOO R²", fontsize=7)
    ax.legend(fontsize=7, framealpha=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.suptitle("Per-protein LOO R² distributions — stacked ensemble",
             fontsize=10, y=1.02, color="#e2e8f0")
plt.tight_layout()
plt.savefig("outputs/reports/residual_distributions.png",
            dpi=150, bbox_inches="tight", facecolor="#05070d")
plt.close()
print("  Saved: residual_distributions.png")

# ── Plot 2 — R² vs protein physicochemical properties ────
print("Plotting R² vs protein features...")
key_features = ["instability_index","isoelectric_point",
                "gravy_score","agg_mean"]
key_features = [f for f in key_features if f in prot.columns]

prot_sub = prot[["protein_id"] + key_features].drop_duplicates("protein_id")
val_feat = val_df.merge(prot_sub, left_on="held_out_protein",
                         right_on="protein_id", how="left")

fig, axes = plt.subplots(len(key_features), len(TARGETS),
                          figsize=(16, 3.5*len(key_features)))
if len(key_features) == 1:
    axes = axes.reshape(1, -1)

for row_idx, feat in enumerate(key_features):
    for col_idx, (target, r2_col) in enumerate(zip(TARGETS, R2_COLS)):
        ax = axes[row_idx, col_idx]
        if feat in val_feat.columns and r2_col in val_feat.columns:
            x = val_feat[feat].values
            y = val_feat[r2_col].values
            mask = ~np.isnan(x) & ~np.isnan(y)
            ax.scatter(x[mask], y[mask], alpha=0.6, s=20,
                       color=COLORS[col_idx], edgecolors="none")
            if mask.sum() > 2:
                z = np.polyfit(x[mask], y[mask], 1)
                x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                ax.plot(x_line, np.poly1d(z)(x_line),
                        color="white", lw=1, alpha=0.7, ls="--")
                corr = np.corrcoef(x[mask], y[mask])[0,1]
                ax.set_title(f"r={corr:.2f}", fontsize=8, pad=4)
        ax.axhline(0, color=ROSE, lw=0.8, alpha=0.5, ls=":")
        ax.set_xlabel(feat.replace("_"," "), fontsize=7)
        if col_idx == 0:
            ax.set_ylabel(target.replace("_score","").replace("_"," "),
                          fontsize=7)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

plt.suptitle("LOO R² vs key protein features — do harder proteins correlate with features?",
             fontsize=9, y=1.01, color="#e2e8f0")
plt.tight_layout()
plt.savefig("outputs/reports/residual_vs_features.png",
            dpi=150, bbox_inches="tight", facecolor="#05070d")
plt.close()
print("  Saved: residual_vs_features.png")

# ── Per-class AUC ─────────────────────────────────────────
print("Computing per-class AUC...")
query_col = None
for c in ["query_label","protein_class","family","category"]:
    if c in prot.columns:
        query_col = c
        break

class_df = pd.DataFrame()
if query_col:
    prot_class = prot[["protein_id",query_col]].drop_duplicates("protein_id")
    val_class  = val_df.merge(prot_class,
                               left_on="held_out_protein",
                               right_on="protein_id", how="left")
    rows = []
    for cls, grp in val_class.groupby(query_col):
        valid = grp["roc_auc"].dropna()
        rows.append({
            "class":          cls,
            "n_proteins":     len(grp),
            "mean_roc_auc":   round(valid.mean(),3) if len(valid)>0 else np.nan,
            "min_roc_auc":    round(valid.min(), 3) if len(valid)>0 else np.nan,
            "mean_comp_r2":   round(grp["composite_r2"].mean(),3),
            "n_single_class": int(grp["roc_auc"].isna().sum()),
        })
    class_df = pd.DataFrame(rows).sort_values("mean_roc_auc")
    class_df.to_csv("outputs/reports/class_auc_summary.csv", index=False)

# ── Bias corrections summary ──────────────────────────────
print(f"\n{'='*55}")
print("RESIDUAL ANALYSIS SUMMARY")
print("="*55)

print(f"\n  LOO bias corrections applied (from bias_corrections.csv):")
for _, row in bias_df.iterrows():
    flag = " <- corrected" if abs(row["bias"]) > 0.001 else " <- near zero"
    print(f"    {row['target']:<25} bias={row['bias']:+.6f}{flag}")

print(f"\n  Per-target LOO R² (post bias correction):")
weakest = None
weakest_r2 = 1.0
for s in target_stats:
    flag = " <- weakest" if s["mean_r2"] == min(t["mean_r2"] for t in target_stats) else ""
    if s["mean_r2"] < weakest_r2:
        weakest_r2 = s["mean_r2"]
        weakest = s["target"]
    print(f"    {s['target']:<25} mean R²={s['mean_r2']:.4f}  "
          f"min R²={s['min_r2']:.4f}{flag}")

print(f"\n  Weakest target: {weakest} (R²={weakest_r2:.4f})")
print(f"  Note: oxidation weakness is a known limitation —")
print(f"  requires AlphaFold SASA for improvement")

if not class_df.empty:
    print(f"\n  Per-class AUC:")
    print(f"  {'Class':<35} {'N':>4} {'Mean AUC':>10} {'Min AUC':>9}")
    print(f"  {'-'*60}")
    for _, r in class_df.iterrows():
        blind = " <- blind spot" if (
            pd.notna(r["mean_roc_auc"]) and r["mean_roc_auc"] < 0.95) else ""
        print(f"  {str(r['class']):<35} {int(r['n_proteins']):>4} "
              f"{str(r['mean_roc_auc']):>10} {str(r['min_roc_auc']):>9}{blind}")

print(f"\nSaved:")
print(f"  outputs/reports/residual_distributions.png")
print(f"  outputs/reports/residual_vs_features.png")
if not class_df.empty:
    print(f"  outputs/reports/class_auc_summary.csv")
print("="*55)
