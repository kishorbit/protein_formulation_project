import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import joblib, os
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

print("\nLoading data...")
prot   = pd.read_csv("data/processed/protein_features.csv")
merged = pd.read_csv("data/processed/dataset_merged.csv")
val    = pd.read_csv("outputs/reports/stacked_validation.csv")

# ── OOD detection features ────────────────────────────────
# Use only the axes where we observed collapse
OOD_FEATURES = ["pct_cys","pct_met","pct_asn","instability_index",
                 "sequence_length","gravy_score","agg_mean","agg_hotspot_frac"]
OOD_FEATURES = [f for f in OOD_FEATURES if f in prot.columns]
print(f"  OOD features: {OOD_FEATURES}")

prot_sub = prot.drop_duplicates("protein_id")
X_prot   = prot_sub[OOD_FEATURES].fillna(prot_sub[OOD_FEATURES].median())

# ── Per-feature sigma-based flags ────────────────────────
print("\nComputing per-feature outlier thresholds (2σ)...")
thresholds = {}
for feat in OOD_FEATURES:
    mu    = X_prot[feat].mean()
    sigma = X_prot[feat].std()
    thresholds[feat] = {"mean": mu, "std": sigma,
                         "lo": mu - 2*sigma, "hi": mu + 2*sigma}

# ── Fit Mahalanobis (EllipticEnvelope) ───────────────────
print("Fitting EllipticEnvelope (Mahalanobis OOD detector)...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_prot)

ee = EllipticEnvelope(contamination=0.05, random_state=42)
ee.fit(X_scaled)
joblib.dump({"scaler": scaler, "detector": ee, "features": OOD_FEATURES,
             "thresholds": thresholds},
            "outputs/models/ood_detector.pkl")
print("  Saved: outputs/models/ood_detector.pkl")

# ── Score all proteins ────────────────────────────────────
scores      = ee.score_samples(X_scaled)   # more negative = more OOD
decisions   = ee.predict(X_scaled)         # -1 = outlier, +1 = inlier

prot_ood = prot_sub[["protein_id"] + OOD_FEATURES].copy()
prot_ood["mahal_score"]  = scores
prot_ood["ood_flag"]     = (decisions == -1).astype(int)

# per-feature sigma flags
for feat in OOD_FEATURES:
    hi = thresholds[feat]["hi"]
    lo = thresholds[feat]["lo"]
    prot_ood[f"outlier_{feat}"] = (
        (prot_ood[feat] > hi) | (prot_ood[feat] < lo)
    ).astype(int)

sigma_flag_cols = [f"outlier_{f}" for f in OOD_FEATURES]
prot_ood["n_sigma_flags"] = prot_ood[sigma_flag_cols].sum(axis=1)

# composite risk
prot_ood["risk_level"] = "low"
prot_ood.loc[prot_ood["ood_flag"] == 1, "risk_level"] = "high"
prot_ood.loc[
    (prot_ood["ood_flag"] == 0) & (prot_ood["n_sigma_flags"] >= 1),
    "risk_level"
] = "medium"

# ── Validate: do OOD flags predict R2 collapse? ───────────
print("\nValidating OOD flags against held-out R2 collapse...")
TARGETS = ["aggregation_score","oxidation_level","deamidation_level",
           "potency_retention","shelf_life_score"]

val_ood = val.merge(
    prot_ood[["protein_id","ood_flag","n_sigma_flags","risk_level","mahal_score"]],
    left_on="held_out_protein", right_on="protein_id", how="left"
)

print(f"\n  {'Risk':<8} {'N':>4} {'ROC-AUC':>9} {'Comp R2':>9} "
      + "  ".join(f"{t.replace('_level','').replace('_score','')[:8]:>8}"
                  for t in TARGETS))
print("  " + "-"*80)
for risk in ["low","medium","high"]:
    grp = val_ood[val_ood["risk_level"] == risk]
    if len(grp) == 0:
        continue
    roc   = grp["roc_auc"].mean()
    comp  = grp["composite_r2"].mean()
    t_r2s = [grp[f"r2_{t}"].mean() for t in TARGETS if f"r2_{t}" in grp.columns]
    print(f"  {risk:<8} {len(grp):>4} {roc:>9.3f} {comp:>9.3f} "
          + "  ".join(f"{v:>8.3f}" if not np.isnan(v) else f"{'nan':>8}"
                      for v in t_r2s))

# Precision/recall of OOD flag for catching R2<0.70
print("\n  OOD flag precision/recall for catching any-target R2 < 0.70:")
val_ood["has_collapse"] = False
for t in TARGETS:
    col = f"r2_{t}"
    if col in val_ood.columns:
        val_ood["has_collapse"] |= (val_ood[col] < 0.70)

tp = ((val_ood["ood_flag"]==1) & val_ood["has_collapse"]).sum()
fp = ((val_ood["ood_flag"]==1) & ~val_ood["has_collapse"]).sum()
fn = ((val_ood["ood_flag"]==0) & val_ood["has_collapse"]).sum()
tn = ((val_ood["ood_flag"]==0) & ~val_ood["has_collapse"]).sum()

prec   = tp / (tp + fp) if (tp+fp) > 0 else 0
recall = tp / (tp + fn) if (tp+fn) > 0 else 0
print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"    Precision={prec:.2f}  Recall={recall:.2f}")

# medium+high combined
val_ood["risk_flag"] = val_ood["risk_level"].isin(["medium","high"])
tp2 = (val_ood["risk_flag"] & val_ood["has_collapse"]).sum()
fp2 = (val_ood["risk_flag"] & ~val_ood["has_collapse"]).sum()
fn2 = (~val_ood["risk_flag"] & val_ood["has_collapse"]).sum()
prec2   = tp2 / (tp2+fp2) if (tp2+fp2) > 0 else 0
recall2 = tp2 / (tp2+fn2) if (tp2+fn2) > 0 else 0
print(f"\n  Medium+High risk flag:")
print(f"    TP={tp2}  FP={fp2}  FN={fn2}")
print(f"    Precision={prec2:.2f}  Recall={recall2:.2f}")

# ── Save scored protein list ──────────────────────────────
out_cols = (["protein_id","risk_level","ood_flag","n_sigma_flags","mahal_score"]
            + sigma_flag_cols + OOD_FEATURES)
prot_ood[out_cols].sort_values("mahal_score").to_csv(
    "outputs/reports/ood_protein_scores.csv", index=False)
print("\n  Saved: outputs/reports/ood_protein_scores.csv")

# ── Plot ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":"#05070d","axes.facecolor":"#0a0e1a",
    "axes.edgecolor":"#1e2535","axes.labelcolor":"#6b748a",
    "xtick.color":"#6b748a","ytick.color":"#6b748a",
    "text.color":"#e2e8f0","grid.color":"#1e2535",
    "grid.linewidth":0.5,"font.size":9,
})
CYAN="#00c8f0"; AMBER="#f0a020"; ROSE="#f05070"; GREEN="#00d68a"

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 1. Mahalanobis score distribution by risk
ax = axes[0]
for risk, color in [("low",GREEN),("medium",AMBER),("high",ROSE)]:
    grp = prot_ood[prot_ood["risk_level"]==risk]["mahal_score"]
    if len(grp):
        ax.hist(grp, bins=20, color=color, alpha=0.7,
                label=f"{risk} (n={len(grp)})", edgecolor="none")
ax.set_title("Mahalanobis score by risk level", fontsize=8)
ax.set_xlabel("Score (more negative = more OOD)", fontsize=7)
ax.legend(fontsize=7, framealpha=0)
ax.grid(alpha=0.2)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# 2. OOD score vs oxidation R2
ax = axes[1]
if "r2_oxidation_level" in val_ood.columns:
    sub = val_ood.dropna(subset=["mahal_score","r2_oxidation_level"])
    colors = [ROSE if r<0 else AMBER if r<0.7 else GREEN
              for r in sub["r2_oxidation_level"]]
    ax.scatter(sub["mahal_score"], sub["r2_oxidation_level"],
               c=colors, alpha=0.8, s=30, edgecolors="none")
    ax.axhline(0.70, color=AMBER, lw=1, ls="--", alpha=0.7)
    ax.axhline(0,    color=ROSE,  lw=1, ls="--", alpha=0.5)
ax.set_title("OOD score vs oxidation R²", fontsize=8)
ax.set_xlabel("Mahalanobis score", fontsize=7)
ax.set_ylabel("oxidation R²", fontsize=7)
ax.grid(alpha=0.2)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# 3. pct_cys vs pct_met, coloured by risk
ax = axes[2]
for risk, color, alpha in [("low",CYAN,0.3),("medium",AMBER,0.7),("high",ROSE,0.9)]:
    grp = prot_ood[prot_ood["risk_level"]==risk]
    if "pct_cys" in grp.columns and "pct_met" in grp.columns:
        ax.scatter(grp["pct_cys"], grp["pct_met"],
                   color=color, alpha=alpha, s=20,
                   label=risk, edgecolors="none")
cys_t = thresholds["pct_cys"]["hi"]
met_t = thresholds["pct_met"]["hi"]
ax.axvline(cys_t, color=AMBER, lw=1, ls="--", alpha=0.6)
ax.axhline(met_t, color=AMBER, lw=1, ls="--", alpha=0.6)
ax.set_title("pct_cys vs pct_met by risk", fontsize=8)
ax.set_xlabel("pct_cys", fontsize=7)
ax.set_ylabel("pct_met", fontsize=7)
ax.legend(fontsize=7, framealpha=0)
ax.grid(alpha=0.2)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.suptitle("OOD detection — protein outlier analysis",
             fontsize=10, y=1.02, color="#e2e8f0")
plt.tight_layout()
plt.savefig("outputs/reports/ood_analysis.png",
            dpi=150, bbox_inches="tight", facecolor="#05070d")
plt.close()
print("  Saved: outputs/reports/ood_analysis.png")

print(f"\n{'='*60}")
print("OOD FLAGGING COMPLETE")
print("="*60)
print(f"""
  Detector: EllipticEnvelope (Mahalanobis distance)
  Features: {OOD_FEATURES}

  At inference time, load outputs/models/ood_detector.pkl
  and call:
    scaler   = ood['scaler']
    detector = ood['detector']
    X_scaled = scaler.transform(protein_features)
    risk     = detector.predict(X_scaled)  # -1 = flag

  Proteins flagged as high-risk should report wider
  prediction intervals and a reliability warning.
""")
print("="*60)
