import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

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
        "temperature_c":    +1, "high_temp_flag":    +1,
        "instability_index":+1, "agg_mean":          +1,
        "agg_hotspot_frac": +1, "sug_conc_mM":       -1,
        "sur_conc_mM":      -1, "sug_x_instability": -1,
    },
    "oxidation_level": {
        "temperature_c":+1, "high_temp_flag":+1,
        "pct_met":      +1, "pct_trp":       +1,
    },
    "deamidation_level": {
        "temperature_c":+1, "high_temp_flag":+1,
        "pct_asn":      +1, "ph":            +1,
    },
    "potency_retention": {
        "temperature_c":    -1, "high_temp_flag":    -1,
        "instability_index":-1, "agg_mean":          -1,
        "sug_conc_mM":      +1,
    },
    "shelf_life_score": {
        "temperature_c":    -1, "high_temp_flag":    -1,
        "instability_index":-1, "sug_conc_mM":       +1,
        "sur_conc_mM":      +1,
    },
}

def build_constraint_tuple(target_name, feature_cols):
    constraints = NAMED_CONSTRAINTS.get(target_name, {})
    return tuple(constraints.get(feat, 0) for feat in feature_cols)

# ── Load data ─────────────────────────────────────────────
print("\nLoading data...")
df = pd.read_csv("data/processed/dataset_merged.csv")
feature_cols = pd.read_csv(
    "data/processed/feature_cols.csv", header=None)[0].tolist()
feature_cols = [c for c in feature_cols if c in df.columns]

proteins = df["protein_id"].unique()
print(f"  Proteins: {len(proteins)}  |  Features: {len(feature_cols)}")

# ── LOO loop — collect all OOS predictions ────────────────
print("\nRunning LOO to collect out-of-sample predictions...")

all_true_targets  = {t: [] for t in REGRESSION_TARGETS}
all_pred_targets  = {t: [] for t in REGRESSION_TARGETS}
all_true_comp     = []
all_pred_comp     = []
all_true_bin      = []
all_pred_signal   = []

for held_out in proteins:
    train_mask = df["protein_id"] != held_out
    test_mask  = df["protein_id"] == held_out

    X_tr = df.loc[train_mask, feature_cols].values
    X_te = df.loc[test_mask,  feature_cols].values
    y_tr = df.loc[train_mask, REGRESSION_TARGETS].values
    y_te = df.loc[test_mask,  REGRESSION_TARGETS].values

    imp = KNNImputer(n_neighbors=5)
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)

    scl = MinMaxScaler()
    X_tr = scl.fit_transform(X_tr)
    X_te = scl.transform(X_te)

    estimators = []
    for i, target in enumerate(REGRESSION_TARGETS):
        xgb = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, tree_method="hist",
            monotone_constraints=build_constraint_tuple(
                target, feature_cols),
        )
        xgb.fit(X_tr, y_tr[:, i])
        estimators.append(xgb)

    y_pred = np.column_stack([e.predict(X_te) for e in estimators])

    comp_pred = (
        COMPOSITE_WEIGHTS["aggregation_score"]  * y_pred[:, 0]
      + COMPOSITE_WEIGHTS["oxidation_level"]    * y_pred[:, 1]
      + COMPOSITE_WEIGHTS["deamidation_level"]  * y_pred[:, 2]
      + COMPOSITE_WEIGHTS["potency_retention"]  * (1 - y_pred[:, 3])
      + COMPOSITE_WEIGHTS["shelf_life_score"]   * (1 - y_pred[:, 4])
    )

    for i, t in enumerate(REGRESSION_TARGETS):
        all_true_targets[t].extend(y_te[:, i].tolist())
        all_pred_targets[t].extend(y_pred[:, i].tolist())

    all_true_comp.extend(
        df.loc[test_mask, "composite_stability_score"].values.tolist())
    all_pred_comp.extend(comp_pred.tolist())
    all_true_bin.extend(
        df.loc[test_mask, "stable"].values.tolist())
    all_pred_signal.extend((1 - comp_pred).tolist())

    print(f"  {held_out} done")

all_true_comp   = np.array(all_true_comp)
all_pred_comp   = np.array(all_pred_comp)
all_true_bin    = np.array(all_true_bin)
all_pred_signal = np.array(all_pred_signal)

# ─────────────────────────────────────────────────────────
# CALIBRATION PLOTS
# ─────────────────────────────────────────────────────────
print("\nGenerating calibration plots...")

plt.style.use("dark_background")
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#05070d")
gs  = gridspec.GridSpec(3, 3, figure=fig,
                         hspace=0.45, wspace=0.35)

CYAN   = "#00c8f0"
GREEN  = "#00d68a"
AMBER  = "#f0a020"
VIOLET = "#7c6af7"
ROSE   = "#f05070"
MUTED  = "#6b748a"
TARGET_COLORS = [ROSE, AMBER, VIOLET, GREEN, CYAN]

# ── Panel 1: Composite score reliability diagram ──────────
ax1 = fig.add_subplot(gs[0, :2])

n_bins = 10
bins   = np.linspace(0, 1, n_bins + 1)
bin_centers, bin_means_true, bin_means_pred = [], [], []

for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (all_pred_comp >= lo) & (all_pred_comp < hi)
    if mask.sum() >= 5:
        bin_centers.append((lo + hi) / 2)
        bin_means_pred.append(all_pred_comp[mask].mean())
        bin_means_true.append(all_true_comp[mask].mean())

bin_centers  = np.array(bin_centers)
bin_means_pred = np.array(bin_means_pred)
bin_means_true = np.array(bin_means_true)

ax1.plot([0, 1], [0, 1], "--", color=MUTED, lw=1.2,
         label="Perfect calibration")
ax1.plot(bin_means_pred, bin_means_true,
         "o-", color=CYAN, lw=2, markersize=7,
         markerfacecolor="#05070d", markeredgewidth=2,
         label="Model calibration")

# Shade calibration gap
ax1.fill_between(bin_means_pred, bin_means_pred, bin_means_true,
                 alpha=0.12, color=CYAN)

# Calibration error
cal_error = np.mean(np.abs(bin_means_pred - bin_means_true))
ax1.set_title(f"Composite Score Reliability Diagram  "
              f"(Mean Cal. Error = {cal_error:.3f})",
              color="white", fontsize=11, pad=10)
ax1.set_xlabel("Predicted composite score", color=MUTED)
ax1.set_ylabel("Observed composite score", color=MUTED)
ax1.legend(fontsize=9, framealpha=0)
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.set_facecolor("#0a0e1a")
ax1.tick_params(colors=MUTED)
for sp in ax1.spines.values():
    sp.set_edgecolor("#1e2535")

# Calibration verdict text
if cal_error < 0.05:
    verdict = "Well calibrated"
    vcol    = GREEN
elif cal_error < 0.10:
    verdict = "Acceptable calibration"
    vcol    = AMBER
else:
    verdict = "Miscalibrated — scores unreliable"
    vcol    = ROSE

ax1.text(0.98, 0.04, verdict, ha="right", va="bottom",
         transform=ax1.transAxes, color=vcol,
         fontsize=10, fontweight="bold")

# ── Panel 2: Prediction interval coverage ────────────────
ax2 = fig.add_subplot(gs[0, 2])

residuals  = all_pred_comp - all_true_comp
rmse_comp  = np.sqrt(np.mean(residuals**2))
within_1sd = np.mean(np.abs(residuals) <= rmse_comp) * 100
within_2sd = np.mean(np.abs(residuals) <= 2 * rmse_comp) * 100

bars = ax2.bar(["±1 RMSE\n(expect 68%)", "±2 RMSE\n(expect 95%)"],
               [within_1sd, within_2sd],
               color=[CYAN, GREEN], alpha=0.8,
               width=0.5, edgecolor="none")
ax2.axhline(68, color=MUTED, lw=1, ls="--", alpha=0.6)
ax2.axhline(95, color=MUTED, lw=1, ls="--", alpha=0.6)

for bar, val in zip(bars, [within_1sd, within_2sd]):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f"{val:.1f}%", ha="center",
             color="white", fontsize=11, fontweight="bold")

ax2.set_title("Interval Coverage", color="white",
              fontsize=11, pad=10)
ax2.set_ylabel("% predictions within interval", color=MUTED)
ax2.set_ylim(0, 110)
ax2.set_facecolor("#0a0e1a")
ax2.tick_params(colors=MUTED)
for sp in ax2.spines.values():
    sp.set_edgecolor("#1e2535")

# ── Panels 3–7: Per-target reliability diagrams ───────────
for idx, (target, color) in enumerate(
        zip(REGRESSION_TARGETS, TARGET_COLORS)):
    row = 1 + idx // 3
    col = idx  % 3
    ax  = fig.add_subplot(gs[row, col])

    t_true = np.array(all_true_targets[target])
    t_pred = np.array(all_pred_targets[target])

    bct, bmt, bmp = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (t_pred >= lo) & (t_pred < hi)
        if mask.sum() >= 3:
            bct.append((lo+hi)/2)
            bmp.append(t_pred[mask].mean())
            bmt.append(t_true[mask].mean())

    if bmt:
        ax.plot([0,1],[0,1],"--", color=MUTED, lw=1, alpha=0.6)
        ax.plot(bmp, bmt, "o-", color=color, lw=1.8,
                markersize=5, markerfacecolor="#05070d",
                markeredgewidth=2)
        ax.fill_between(bmp, bmp, bmt, alpha=0.1, color=color)
        ce = np.mean(np.abs(np.array(bmp) - np.array(bmt)))
        r2 = r2_score(t_true, t_pred)
        ax.set_title(
            f"{target.replace('_',' ').title()}\n"
            f"R²={r2:.3f}  CalErr={ce:.3f}",
            color="white", fontsize=9, pad=6)
    else:
        ax.set_title(target, color="white", fontsize=9)

    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Predicted", color=MUTED, fontsize=8)
    ax.set_ylabel("Observed", color=MUTED, fontsize=8)
    ax.set_facecolor("#0a0e1a")
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2535")

# ── Summary stats ─────────────────────────────────────────
comp_r2  = r2_score(all_true_comp, all_pred_comp)
roc_auc  = roc_auc_score(all_true_bin, all_pred_signal)
pr_auc   = average_precision_score(all_true_bin, all_pred_signal)

fig.suptitle(
    f"FormulAI — Calibration Report  |  "
    f"Composite R²={comp_r2:.3f}  "
    f"ROC-AUC={roc_auc:.3f}  "
    f"PR-AUC={pr_auc:.3f}  "
    f"RMSE={rmse_comp:.3f}",
    color="white", fontsize=12, y=0.98)

plt.savefig("outputs/reports/calibration_report.png",
            dpi=150, bbox_inches="tight",
            facecolor="#05070d")
plt.close()

# ── Console summary ───────────────────────────────────────
print(f"\n{'='*60}")
print("CALIBRATION REPORT — Out-of-sample (LOO)")
print("="*60)
print(f"\nComposite score calibration:")
print(f"  Mean calibration error : {cal_error:.4f}  → {verdict}")
print(f"  RMSE (composite)       : {rmse_comp:.4f}")
print(f"  R² (composite)         : {comp_r2:.4f}")
print(f"  Coverage at ±1 RMSE    : {within_1sd:.1f}%  (expect ~68%)")
print(f"  Coverage at ±2 RMSE    : {within_2sd:.1f}%  (expect ~95%)")

print(f"\nPer-target calibration:")
for target in REGRESSION_TARGETS:
    t_true = np.array(all_true_targets[target])
    t_pred = np.array(all_pred_targets[target])
    r2     = r2_score(t_true, t_pred)
    rmse   = np.sqrt(mean_squared_error(t_true, t_pred))
    bias   = np.mean(t_pred - t_true)
    print(f"  {target:<25}  R²={r2:.3f}  "
          f"RMSE={rmse:.3f}  Bias={bias:+.3f}")

print(f"\nDerived classification:")
print(f"  ROC-AUC : {roc_auc:.4f}")
print(f"  PR-AUC  : {pr_auc:.4f}")
print(f"\nSaved: outputs/reports/calibration_report.png")
print("="*60)
