import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import joblib, os
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/models/calibrators", exist_ok=True)

print("\nLoading data...")
df = pd.read_csv("outputs/reports/excipient_recommendations.csv")
merged = pd.read_csv("data/processed/dataset_merged.csv")
print(f"  Recommendations: {df.shape}, Merged: {merged.shape}")

TARGETS = ["aggregation_score", "oxidation_level", "deamidation_level",
           "potency_retention", "shelf_life_score"]
PRED    = ["pred_aggregation",  "pred_oxidation",  "pred_deamidation",
           "pred_potency",      "pred_shelf_life"]

pairs = [(p, t) for p, t in zip(PRED, TARGETS)
         if p in df.columns and t in df.columns]
print(f"  Found {len(pairs)} target pairs for calibration")

print("\nFitting isotonic calibrators (5-fold OOF CV)...")

records_pre  = []
records_post = []
calibrators  = {}

for pred_col, true_col in pairs:
    y_raw  = df[pred_col].values
    y_true = df[true_col].values
    mask   = ~np.isnan(y_raw) & ~np.isnan(y_true)
    y_raw_m, y_true_m = y_raw[mask], y_true[mask]

    iso_cv = IsotonicRegression(out_of_bounds="clip")
    y_cal_oof = cross_val_predict(
        iso_cv, y_raw_m.reshape(-1, 1), y_true_m,
        cv=5, method="predict"
    ).clip(0, 1)

    iso_final = IsotonicRegression(out_of_bounds="clip")
    iso_final.fit(y_raw_m, y_true_m)
    calibrators[true_col] = iso_final
    joblib.dump(iso_final,
        f"outputs/models/calibrators/iso_{true_col}.pkl")

    pre_bias  = (y_raw_m   - y_true_m).mean()
    post_bias = (y_cal_oof - y_true_m).mean()
    pre_r2    = r2_score(y_true_m, y_raw_m)
    post_r2   = r2_score(y_true_m, y_cal_oof)
    pre_mae   = mean_absolute_error(y_true_m, y_raw_m)
    post_mae  = mean_absolute_error(y_true_m, y_cal_oof)

    records_pre.append( {"target": true_col, "bias": pre_bias,
                          "r2": pre_r2, "mae": pre_mae})
    records_post.append({"target": true_col, "bias": post_bias,
                          "r2": post_r2, "mae": post_mae})

    print(f"\n  {true_col}")
    print(f"    Pre-cal : bias={pre_bias:+.4f}  R²={pre_r2:.4f}  MAE={pre_mae:.4f}")
    print(f"    Post-cal: bias={post_bias:+.4f}  R²={post_r2:.4f}  MAE={post_mae:.4f}")
    delta = abs(pre_bias) - abs(post_bias)
    print(f"    {'✓' if delta > 0 else '✗'} Bias reduced by {delta:+.4f}")

# ── Write cal_* columns back to excipient_recommendations.csv ──
print("\nWriting calibrated predictions...")
df_out = df.copy()
for pred_col, true_col in pairs:
    y_raw = df_out[pred_col].values
    mask  = ~np.isnan(y_raw)
    cal_col = pred_col.replace("pred_", "cal_")
    df_out[cal_col] = np.nan
    df_out.loc[mask, cal_col] = (
        calibrators[true_col].predict(y_raw[mask]).clip(0, 1)
    )
df_out.to_csv("outputs/reports/excipient_recommendations.csv", index=False)
print("  excipient_recommendations.csv updated with cal_* columns")

# ── Also join cal_* into dataset_merged.csv via protein_id ──
cal_cols = [pred_col.replace("pred_", "cal_") for pred_col, _ in pairs]
merge_patch = df_out[["protein_id"] + cal_cols].drop_duplicates("protein_id")
merged_out  = merged.merge(merge_patch, on="protein_id", how="left")
merged_out.to_csv("data/processed/dataset_merged.csv", index=False)
print("  dataset_merged.csv patched with cal_* columns via protein_id")

# ── Summary table ─────────────────────────────────────────
df_pre  = pd.DataFrame(records_pre ).set_index("target")
df_post = pd.DataFrame(records_post).set_index("target")
summary = pd.DataFrame({
    "bias_pre":   df_pre ["bias"].round(4),
    "bias_post":  df_post["bias"].round(4),
    "bias_delta": (df_pre["bias"].abs() - df_post["bias"].abs()).round(4),
    "r2_pre":     df_pre ["r2"].round(4),
    "r2_post":    df_post["r2"].round(4),
    "mae_pre":    df_pre ["mae"].round(4),
    "mae_post":   df_post["mae"].round(4),
    "mae_delta":  (df_pre["mae"] - df_post["mae"]).round(4),
})
summary.to_csv("outputs/reports/calibration_summary.csv")

# ── Plot 1: scatter raw vs calibrated ─────────────────────
plt.rcParams.update({
    "figure.facecolor": "#05070d", "axes.facecolor": "#0a0e1a",
    "axes.edgecolor":   "#1e2535", "axes.labelcolor": "#6b748a",
    "xtick.color":      "#6b748a", "ytick.color":     "#6b748a",
    "text.color":       "#e2e8f0", "grid.color":      "#1e2535",
    "grid.linewidth": 0.5, "font.size": 9,
})
ROSE = "#f05070"; GREEN = "#00d68a"

fig, axes = plt.subplots(1, len(pairs), figsize=(16, 3.8))
if len(pairs) == 1:
    axes = [axes]

for ax, (pred_col, true_col) in zip(axes, pairs):
    y_raw  = df[pred_col].values
    y_true = df[true_col].values
    mask   = ~np.isnan(y_raw) & ~np.isnan(y_true)
    y_raw_m, y_true_m = y_raw[mask], y_true[mask]
    y_cal  = calibrators[true_col].predict(y_raw_m).clip(0, 1)

    ax.scatter(y_true_m, y_raw_m, alpha=0.15, s=5,
               color=ROSE,  label="raw",        edgecolors="none")
    ax.scatter(y_true_m, y_cal,   alpha=0.30, s=5,
               color=GREEN, label="calibrated", edgecolors="none")
    lo = min(y_true_m.min(), y_raw_m.min())
    hi = max(y_true_m.max(), y_raw_m.max())
    ax.plot([lo, hi], [lo, hi], color="white", lw=1, alpha=0.5, ls="--")

    b_pre  = (y_raw_m - y_true_m).mean()
    b_post = (y_cal   - y_true_m).mean()
    ax.set_title(f"{true_col.replace('_',' ')}\nbias {b_pre:+.3f} → {b_post:+.3f}",
                 fontsize=8, pad=5)
    ax.set_xlabel("True", fontsize=7)
    ax.set_ylabel("Predicted", fontsize=7)
    ax.legend(fontsize=6, framealpha=0, markerscale=2)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("Isotonic calibration — raw vs calibrated predictions",
             fontsize=10, y=1.02, color="#e2e8f0")
plt.tight_layout()
plt.savefig("outputs/reports/calibration_scatter.png",
            dpi=150, bbox_inches="tight", facecolor="#05070d")
plt.close()
print("  Saved: calibration_scatter.png")

# ── Plot 2: ISO mapping curves ────────────────────────────
COLORS = ["#f05070", "#f0a020", "#7c6af7", "#00d68a", "#00c8f0"]
fig, axes = plt.subplots(1, len(pairs), figsize=(16, 3.5))
if len(pairs) == 1:
    axes = [axes]

for ax, (pred_col, true_col), color in zip(axes, pairs, COLORS):
    iso    = calibrators[true_col]
    x_line = np.linspace(0, 1, 300)
    y_line = iso.predict(x_line)
    ax.plot(x_line, y_line, color=color, lw=2, label="ISO curve")
    ax.plot([0, 1], [0, 1], color="white", lw=1, ls="--", alpha=0.4,
            label="perfect cal")
    ax.set_title(true_col.replace("_", " "), fontsize=8, pad=5)
    ax.set_xlabel("Raw prediction", fontsize=7)
    ax.set_ylabel("Calibrated",     fontsize=7)
    ax.legend(fontsize=6, framealpha=0)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("Isotonic calibration curves",
             fontsize=10, y=1.02, color="#e2e8f0")
plt.tight_layout()
plt.savefig("outputs/reports/calibration_curves.png",
            dpi=150, bbox_inches="tight", facecolor="#05070d")
plt.close()
print("  Saved: calibration_curves.png")

# ── Final print ───────────────────────────────────────────
print(f"\n{'='*68}")
print("CALIBRATION SUMMARY")
print("="*68)
print(f"\n  {'Target':<25} {'Bias Pre':>9} {'Bias Post':>10} "
      f"{'R² Pre':>8} {'R² Post':>9} {'MAE Δ':>8}")
print(f"  {'-'*73}")
all_improved = True
for tgt in summary.index:
    r    = summary.loc[tgt]
    flag = "✓" if abs(r.bias_post) < abs(r.bias_pre) else "✗"
    if flag == "✗":
        all_improved = False
    print(f"  {flag} {tgt:<23} {r.bias_pre:>+9.4f} {r.bias_post:>+10.4f} "
          f"{r.r2_pre:>8.4f} {r.r2_post:>9.4f} {r.mae_delta:>+8.4f}")
print()
print("  All targets improved. ✓" if all_improved else
      "  Warning: some targets did not improve.")
print(f"""
  Outputs:
    outputs/models/calibrators/iso_<target>.pkl
    outputs/reports/calibration_summary.csv
    outputs/reports/calibration_scatter.png
    outputs/reports/calibration_curves.png
""")
print("="*68)
