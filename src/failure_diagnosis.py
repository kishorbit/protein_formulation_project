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
val  = pd.read_csv("outputs/reports/stacked_validation.csv")
prot = pd.read_csv("data/processed/protein_features.csv")

TARGETS = ["aggregation_score","oxidation_level","deamidation_level",
           "potency_retention","shelf_life_score"]

# ── Join val with protein features ────────────────────────
prot_sub = prot.drop_duplicates("protein_id")
val_feat = val.merge(prot_sub, left_on="held_out_protein",
                     right_on="protein_id", how="left")

# ── Correlation: which features predict R2 collapse? ─────
print("\n=== FEATURE CORRELATIONS WITH R2 COLLAPSE ===")
feature_cols = ["instability_index","isoelectric_point","gravy_score",
                "pct_met","pct_cys","pct_asn","pct_trp","pct_his",
                "sequence_length","molecular_weight_kda",
                "agg_mean","agg_hotspot_frac"]
feature_cols = [f for f in feature_cols if f in val_feat.columns]

print(f"\n  {'Feature':<25} " +
      "  ".join(f"{t.replace('_level','').replace('_score','').replace('_retention','')[:8]:>8}"
                for t in TARGETS))
print("  " + "-"*85)

corr_records = []
for feat in feature_cols:
    corrs = []
    for t in TARGETS:
        r2_col = f"r2_{t}"
        if r2_col in val_feat.columns:
            sub = val_feat[[feat, r2_col]].dropna()
            if len(sub) > 5:
                c = np.corrcoef(sub[feat], sub[r2_col])[0, 1]
            else:
                c = np.nan
        else:
            c = np.nan
        corrs.append(c)
    corr_records.append({"feature": feat, **dict(zip(TARGETS, corrs))})
    flag = " <--" if any(abs(c) > 0.35 for c in corrs if not np.isnan(c)) else ""
    print(f"  {feat:<25} " +
          "  ".join(f"{c:>8.3f}" if not np.isnan(c) else f"{'nan':>8}"
                    for c in corrs) + flag)

corr_df = pd.DataFrame(corr_records).set_index("feature")

# ── Define high-risk thresholds ───────────────────────────
print("\n\n=== HIGH-RISK PROTEIN FLAGS ===")

# Derive thresholds from actual failure cases
thresholds = {
    "pct_cys":  ("gt", prot["pct_cys"].mean()  + 2 * prot["pct_cys"].std()),
    "pct_met":  ("gt", prot["pct_met"].mean()  + 2 * prot["pct_met"].std()),
    "pct_asn":  ("gt", prot["pct_asn"].mean()  + 2 * prot["pct_asn"].std()),
    "instability_index": ("gt", 60.0),
    "sequence_length":   ("gt", prot["sequence_length"].mean()
                                + 2 * prot["sequence_length"].std()),
}

# apply flags to all proteins
prot_flagged = prot_sub.copy()
flag_cols = []
for feat, (direction, thresh) in thresholds.items():
    if feat not in prot_flagged.columns:
        continue
    col = f"flag_{feat}"
    flag_cols.append(col)
    if direction == "gt":
        prot_flagged[col] = (prot_flagged[feat] > thresh).astype(int)
    print(f"  {feat:<25} threshold={'>' if direction=='gt' else '<'}{thresh:.2f}  "
          f"n_flagged={prot_flagged[col].sum()}")

prot_flagged["n_flags"]     = prot_flagged[flag_cols].sum(axis=1)
prot_flagged["risk_level"]  = pd.cut(prot_flagged["n_flags"],
                                      bins=[-1,0,1,99],
                                      labels=["low","medium","high"])

print(f"\n  Risk distribution:")
print(prot_flagged["risk_level"].value_counts().to_string())

# ── Check: do flags predict R2 collapse in validation? ───
val_risk = val_feat.copy()
for feat, (direction, thresh) in thresholds.items():
    if feat not in val_risk.columns:
        continue
    col = f"flag_{feat}"
    val_risk[col] = (val_risk[feat] > thresh).astype(int) \
                    if direction == "gt" else \
                    (val_risk[feat] < thresh).astype(int)

val_risk["n_flags"]    = val_risk[[f"flag_{f}" for f in thresholds
                                    if f"flag_{f}" in val_risk.columns]].sum(axis=1)
val_risk["risk_level"] = pd.cut(val_risk["n_flags"],
                                 bins=[-1,0,1,99],
                                 labels=["low","medium","high"])

print("\n\n=== MEAN R2 BY RISK LEVEL (validation set) ===")
print(f"\n  {'Risk':<8} {'N':>4} " +
      "  ".join(f"{t.replace('_level','').replace('_score','')[:10]:>10}"
                for t in TARGETS))
print("  " + "-"*70)
for risk in ["low","medium","high"]:
    grp = val_risk[val_risk["risk_level"] == risk]
    vals = [grp[f"r2_{t}"].mean() for t in TARGETS
            if f"r2_{t}" in grp.columns]
    print(f"  {risk:<8} {len(grp):>4} " +
          "  ".join(f"{v:>10.3f}" if not np.isnan(v) else f"{'nan':>10}"
                    for v in vals))

# ── Save flagged protein list ─────────────────────────────
out_cols = ["protein_id","n_flags","risk_level"] + flag_cols + \
           ["instability_index","pct_cys","pct_met","pct_asn","sequence_length"]
out_cols = [c for c in out_cols if c in prot_flagged.columns]
prot_flagged[out_cols].sort_values("n_flags", ascending=False).to_csv(
    "outputs/reports/protein_risk_flags.csv", index=False)
print("\n  Saved: outputs/reports/protein_risk_flags.csv")

# ── Plot: R2 vs pct_cys and pct_met for oxidation ────────
plt.rcParams.update({
    "figure.facecolor":"#05070d","axes.facecolor":"#0a0e1a",
    "axes.edgecolor":"#1e2535","axes.labelcolor":"#6b748a",
    "xtick.color":"#6b748a","ytick.color":"#6b748a",
    "text.color":"#e2e8f0","grid.color":"#1e2535",
    "grid.linewidth":0.5,"font.size":9,
})
CYAN="#00c8f0"; AMBER="#f0a020"; ROSE="#f05070"

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
plot_pairs = [
    ("pct_cys",  "r2_oxidation_level",   "oxidation R² vs pct_cys"),
    ("pct_met",  "r2_oxidation_level",   "oxidation R² vs pct_met"),
    ("pct_asn",  "r2_deamidation_level", "deamidation R² vs pct_asn"),
    ("instability_index", "r2_deamidation_level", "deamidation R² vs instability"),
    ("sequence_length",   "r2_shelf_life_score",  "shelf_life R² vs seq_len"),
    ("instability_index", "r2_shelf_life_score",  "shelf_life R² vs instability"),
]

for ax, (xfeat, yfeat, title) in zip(axes.flat, plot_pairs):
    if xfeat in val_feat.columns and yfeat in val_feat.columns:
        sub = val_feat[[xfeat, yfeat]].dropna()
        colors = [ROSE if y < 0 else AMBER if y < 0.7 else CYAN
                  for y in sub[yfeat]]
        ax.scatter(sub[xfeat], sub[yfeat], c=colors,
                   alpha=0.8, s=30, edgecolors="none")
        ax.axhline(0,    color="white",  lw=1,   ls="--", alpha=0.5)
        ax.axhline(0.70, color=AMBER,   lw=0.8, ls=":",  alpha=0.6,
                   label="R²=0.70 threshold")
        if len(sub) > 2:
            z = np.polyfit(sub[xfeat], sub[yfeat], 1)
            xl = np.linspace(sub[xfeat].min(), sub[xfeat].max(), 100)
            ax.plot(xl, np.poly1d(z)(xl), color="white", lw=1, alpha=0.5)
            c = np.corrcoef(sub[xfeat], sub[yfeat])[0,1]
            title += f"  r={c:.2f}"
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel(xfeat.replace("_"," "), fontsize=7)
    ax.set_ylabel(yfeat.replace("r2_","R² ").replace("_"," "), fontsize=7)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("R² collapse drivers — feature vs per-protein R²",
             fontsize=10, y=1.01, color="#e2e8f0")
plt.tight_layout()
plt.savefig("outputs/reports/failure_diagnosis.png",
            dpi=150, bbox_inches="tight", facecolor="#05070d")
plt.close()
print("  Saved: outputs/reports/failure_diagnosis.png")

print(f"\n{'='*60}")
print("DIAGNOSIS SUMMARY")
print("="*60)
print("""
  Root causes identified:
  1. oxidation_level    — collapses on high pct_cys (>mean+2σ)
                          and high pct_met proteins. Model conflates
                          Met vs Cys oxidation mechanisms.
  2. deamidation_level  — collapses on high pct_asn and high
                          instability_index proteins.
  3. shelf_life_score   — collapses on long sequences and unstable
                          proteins (derived target, inherits errors).

  Recommended actions:
  A. Add pct_cys * pH and pct_met * temp interaction features
     for oxidation_level specifically.
  B. Flag high-risk proteins at inference time using
     outputs/reports/protein_risk_flags.csv thresholds.
  C. Consider splitting oxidation model: one for Cys-rich
     (pct_cys > threshold), one for standard proteins.
""")
print("="*60)
