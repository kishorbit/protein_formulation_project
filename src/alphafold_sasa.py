import pandas as pd
import numpy as np
import urllib.request
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import freesasa
from Bio.PDB import PDBParser

os.makedirs("data/processed/alphafold_pdbs", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

MAX_SASA = {
    "ALA":129.0,"ARG":274.0,"ASN":195.0,"ASP":193.0,
    "CYS":167.0,"GLN":223.0,"GLU":223.0,"GLY":104.0,
    "HIS":224.0,"ILE":197.0,"LEU":201.0,"LYS":236.0,
    "MET":224.0,"PHE":240.0,"PRO":159.0,"SER":155.0,
    "THR":172.0,"TRP":285.0,"TYR":263.0,"VAL":174.0,
}
EXPOSED = 0.25

def get_pdb_url(uniprot_id):
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())[0]["pdbUrl"]

def download_alphafold_pdb(uniprot_id, out_dir):
    pdb_path = os.path.join(out_dir, f"{uniprot_id}.pdb")
    if os.path.exists(pdb_path):
        return pdb_path
    try:
        urllib.request.urlretrieve(get_pdb_url(uniprot_id), pdb_path)
        return pdb_path
    except Exception as e:
        print(f"    Download failed {uniprot_id}: {e}")
        return None

def compute_sasa_features(pdb_path, uniprot_id):
    try:
        structure_fs = freesasa.Structure(pdb_path)
        result       = freesasa.calc(structure_fs)
        areas        = result.residueAreas()  # {chain: {res_str: ResidueArea}}

        # Build lookup: res_id (int) -> (resname, total_sasa)
        parser  = PDBParser(QUIET=True)
        struct  = parser.get_structure(uniprot_id, pdb_path)
        chain   = list(struct[0].get_chains())[0]
        chain_id = chain.get_id()

        chain_areas = areas.get(chain_id, {})

        met_rsa, trp_rsa = [], []

        for res in chain.get_residues():
            aa     = res.get_resname()
            if aa not in ("MET", "TRP"):
                continue
            res_id = str(res.get_id()[1])
            ra = chain_areas.get(res_id)
            if ra is None:
                continue
            total = ra.total          # Å² total SASA for this residue
            max_s = MAX_SASA.get(aa, 200.0)
            rsa   = min(total / max_s, 1.0) if max_s > 0 else 0.0
            if aa == "MET":
                met_rsa.append(rsa)
            else:
                trp_rsa.append(rsa)

        combined = met_rsa + trp_rsa
        return {
            "protein_id":            uniprot_id,
            "met_rsa_mean":          np.mean(met_rsa)  if met_rsa    else 0.0,
            "met_rsa_max":           np.max(met_rsa)   if met_rsa    else 0.0,
            "met_exposed_fraction":  np.mean([r>EXPOSED for r in met_rsa]) if met_rsa else 0.0,
            "met_exposed_count":     sum(1 for r in met_rsa if r>EXPOSED),
            "met_buried_fraction":   np.mean([r<=EXPOSED for r in met_rsa]) if met_rsa else 1.0,
            "trp_rsa_mean":          np.mean(trp_rsa)  if trp_rsa    else 0.0,
            "trp_rsa_max":           np.max(trp_rsa)   if trp_rsa    else 0.0,
            "trp_exposed_fraction":  np.mean([r>EXPOSED for r in trp_rsa]) if trp_rsa else 0.0,
            "trp_exposed_count":     sum(1 for r in trp_rsa if r>EXPOSED),
            "trp_buried_fraction":   np.mean([r<=EXPOSED for r in trp_rsa]) if trp_rsa else 1.0,
            "ox_exposed_count":      sum(1 for r in combined if r>EXPOSED),
            "ox_mean_rsa":           np.mean(combined) if combined else 0.0,
            "ox_max_rsa":            np.max(combined)  if combined else 0.0,
            "n_met":                 len(met_rsa),
            "n_trp":                 len(trp_rsa),
        }
    except Exception as e:
        print(f"    SASA failed {uniprot_id}: {e}")
        return None

# ── Main ──────────────────────────────────────────────────
print("\nLoading protein list...")
prot_df = pd.read_csv("data/processed/protein_features.csv")
ids     = prot_df["protein_id"].unique()
print(f"  Proteins: {len(ids)}")

print(f"\n  {'Protein':<14} {'Met RSA':>8} {'Trp RSA':>8} {'Met exp':>8} {'Trp exp':>8} {'nM':>4} {'nW':>4}")
print(f"  {'-'*58}")

all_features, failed = [], []

for uid in ids:
    pdb_path = download_alphafold_pdb(uid, "data/processed/alphafold_pdbs")
    if pdb_path is None:
        failed.append(uid); continue

    feats = compute_sasa_features(pdb_path, uid)
    if feats is None:
        failed.append(uid); continue

    all_features.append(feats)
    print(f"  {uid:<14} {feats['met_rsa_mean']:>8.3f} {feats['trp_rsa_mean']:>8.3f} "
          f"{feats['met_exposed_fraction']:>8.3f} {feats['trp_exposed_fraction']:>8.3f} "
          f"{feats['n_met']:>4} {feats['n_trp']:>4}")
    time.sleep(0.1)

sasa_df = pd.DataFrame(all_features)
sasa_df.to_csv("outputs/reports/alphafold_sasa_features.csv", index=False)

print(f"\n{'='*55}")
print("ALPHAFOLD SASA COMPLETE")
print("="*55)
print(f"  Processed: {len(all_features)}/40   Failed: {len(failed)}")
for col in ["met_rsa_mean","met_exposed_fraction","trp_rsa_mean","trp_exposed_fraction","ox_mean_rsa"]:
    if col in sasa_df.columns:
        print(f"  {col:<25} mean={sasa_df[col].mean():.3f}  std={sasa_df[col].std():.3f}")
print(f"\nSaved: outputs/reports/alphafold_sasa_features.csv")
print("="*55)
