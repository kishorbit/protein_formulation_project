import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.covariance import EmpiricalCovariance

print("\nLoading data...")
df = pd.read_csv("data/processed/dataset_merged.csv")
feature_cols = pd.read_csv(
    "data/processed/feature_cols.csv", header=None)[0].tolist()
feature_cols = [c for c in feature_cols if c in df.columns]

print(f"  Dataset shape:   {df.shape}")
print(f"  Feature columns: {len(feature_cols)}")

# ── Fit imputer + scaler on full training data ────────────
X_raw = df[feature_cols].values
imputer = KNNImputer(n_neighbors=5)
X = imputer.fit_transform(X_raw)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# ── Fit Mahalanobis covariance on training distribution ───
print("\nFitting Mahalanobis covariance estimator...")
cov = EmpiricalCovariance()
cov.fit(X)

# ── Compute Mahalanobis distances for all training points ─
distances = cov.mahalanobis(X)
distances = np.sqrt(distances)

print(f"  Training distribution distances:")
print(f"    Mean:   {distances.mean():.3f}")
print(f"    Std:    {distances.std():.3f}")
print(f"    95th p: {np.percentile(distances, 95):.3f}")
print(f"    99th p: {np.percentile(distances, 99):.3f}")

threshold_95 = np.percentile(distances, 95)
threshold_99 = np.percentile(distances, 99)

# ── Flag training samples as in/out of distribution ───────
ood_flags = distances > threshold_95
print(f"\n  Samples flagged as OOD (>95th percentile): "
      f"{ood_flags.sum()} / {len(distances)} "
      f"({100*ood_flags.mean():.1f}%)")

# ── Per-protein OOD summary ───────────────────────────────
df["mahal_distance"]  = distances
df["ood_flag"]        = ood_flags.astype(int)
df["ood_severity"]    = np.where(distances > threshold_99, "High",
                        np.where(distances > threshold_95, "Medium", "In-distribution"))

protein_ood = df.groupby("protein_id").agg(
    mean_distance   = ("mahal_distance","mean"),
    max_distance    = ("mahal_distance","max"),
    pct_ood         = ("ood_flag","mean"),
    ood_severity    = ("ood_severity", lambda x: x.value_counts().index[0])
).reset_index()
protein_ood["mean_distance"] = protein_ood["mean_distance"].round(3)
protein_ood["max_distance"]  = protein_ood["max_distance"].round(3)
protein_ood["pct_ood"]       = protein_ood["pct_ood"].round(3)
protein_ood = protein_ood.sort_values("mean_distance", ascending=False)

protein_ood.to_csv("outputs/reports/ood_report.csv", index=False)

print(f"\n{'='*55}")
print("OOD DETECTION SUMMARY")
print("="*55)
print(f"  95th percentile threshold: {threshold_95:.3f}")
print(f"  99th percentile threshold: {threshold_99:.3f}")
print(f"\n  Top 10 most OOD proteins:")
print(f"  {'Protein':<12} {'Mean dist':>10} {'Max dist':>10} "
      f"{'% OOD':>8} {'Severity':>14}")
print(f"  {'-'*58}")
for _, r in protein_ood.head(10).iterrows():
    print(f"  {r['protein_id']:<12} {r['mean_distance']:>10.3f} "
          f"{r['max_distance']:>10.3f} {r['pct_ood']*100:>7.1f}% "
          f"{r['ood_severity']:>14}")

print(f"\n  Proteins fully in-distribution: "
      f"{(protein_ood['pct_ood']==0).sum()} / {len(protein_ood)}")

# ── Save detector artifacts ───────────────────────────────
os.makedirs("outputs/models", exist_ok=True)
joblib.dump({
    "imputer":       imputer,
    "scaler":        scaler,
    "cov":           cov,
    "threshold_95":  threshold_95,
    "threshold_99":  threshold_99,
    "feature_cols":  feature_cols,
    "distance_mean": distances.mean(),
    "distance_std":  distances.std(),
}, "outputs/models/ood_detector.pkl")

print(f"\nSaved: outputs/reports/ood_report.csv")
print(f"Saved: outputs/models/ood_detector.pkl")
print("="*55)
