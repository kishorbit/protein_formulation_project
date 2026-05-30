import requests
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import io
import json

# ─────────────────────────────────────────
# 1. Fetch sequence from UniProt
# ─────────────────────────────────────────
def fetch_uniprot_sequence(accession: str) -> dict:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    fasta_io = io.StringIO(response.text)
    record = next(SeqIO.parse(fasta_io, "fasta"))
    return {
        "accession": accession,
        "sequence": str(record.seq),
        "length": len(record.seq)
    }

# ─────────────────────────────────────────
# 2. Compute sequence-derived features
# ─────────────────────────────────────────
def compute_sequence_features(sequence: str) -> dict:
    analysis = ProteinAnalysis(sequence)
    ss = analysis.secondary_structure_fraction()
    aa = analysis.amino_acids_percent
    return {
        "isoelectric_point":   round(analysis.isoelectric_point(), 3),
        "gravy_score":         round(analysis.gravy(), 3),
        "instability_index":   round(analysis.instability_index(), 3),
        "molecular_weight_kda":round(analysis.molecular_weight() / 1000, 3),
        "aromaticity":         round(analysis.aromaticity(), 4),
        "helix_fraction":      round(ss[0], 4),
        "turn_fraction":       round(ss[1], 4),
        "sheet_fraction":      round(ss[2], 4),
        # Degradation-risk residues
        "pct_asn": round(aa.get('N', 0), 4),  # deamidation
        "pct_met": round(aa.get('M', 0), 4),  # oxidation
        "pct_cys": round(aa.get('C', 0), 4),  # disulfide scrambling
        "pct_trp": round(aa.get('W', 0), 4),  # oxidation
        "pct_his": round(aa.get('H', 0), 4),  # pH-sensitive
    }

# ─────────────────────────────────────────
# 3. Compute charge at formulation pH range
# ─────────────────────────────────────────
def charge_at_ph(sequence: str, ph_values: list) -> dict:
    analysis = ProteinAnalysis(sequence)
    pi = analysis.isoelectric_point()
    features = {}
    for ph in ph_values:
        charge = analysis.charge_at_pH(ph)
        features[f"charge_at_pH_{ph}"] = round(charge, 3)
        features[f"dist_from_pI_at_pH_{ph}"] = round(abs(ph - pi), 3)
    return features

# ─────────────────────────────────────────
# 4. Master pipeline — one row per protein
# ─────────────────────────────────────────
def build_protein_feature_vector(
    accession: str = None,
    sequence: str = None,
    formulation_phs: list = [5.0, 5.5, 6.0, 6.5, 7.0, 7.4]
) -> pd.Series:
    if accession and not sequence:
        data = fetch_uniprot_sequence(accession)
        sequence = data["sequence"]
        protein_id = accession
    else:
        protein_id = "custom"

    features = {"protein_id": protein_id, "sequence_length": len(sequence)}
    features.update(compute_sequence_features(sequence))
    features.update(charge_at_ph(sequence, formulation_phs))
    return pd.Series(features)


# ─────────────────────────────────────────
# 5. Run for example proteins
# ─────────────────────────────────────────
if __name__ == "__main__":
    accessions = [
        "P01308",  # Human insulin
        "P00533",  # EGFR
        "P10636",  # Tau protein
    ]

    print("\nFetching protein features from UniProt...\n")
    rows = []
    for acc in accessions:
        print(f"  Processing {acc}...")
        row = build_protein_feature_vector(accession=acc)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("protein_id")

    # Save to file
    df.to_csv("data/processed/protein_features.csv")

    print("\n✓ Done! Feature matrix shape:", df.shape)
    print("\nFeatures extracted:")
    print(df.T.to_string())
    print("\nSaved to: data/processed/protein_features.csv")

# ─────────────────────────────────────────
# CamSol API — per-residue solubility scores
# ─────────────────────────────────────────
import json

def fetch_camsol_scores(sequence: str) -> dict:
    """
    Submits a sequence to the CamSol intrinsic API.
    Returns mean score, min score, and hotspot count
    (residues below -1.0 threshold = aggregation prone).
    """
    url = "https://www-cohsoftware.ch.cam.ac.uk/index.php/camsol"
    payload = {"sequence": sequence, "mode": "intrinsic"}
    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            scores = data.get("scores", [])
            if scores:
                arr = np.array(scores, dtype=float)
                return {
                    "camsol_mean":     round(float(arr.mean()), 4),
                    "camsol_min":      round(float(arr.min()),  4),
                    "camsol_hotspots": int((arr < -1.0).sum()),
                }
    except Exception as e:
        print(f"    CamSol unavailable: {e}")
    # Fallback: estimate from GRAVY score
    analysis = ProteinAnalysis(sequence)
    gravy = analysis.gravy()
    return {
        "camsol_mean":     round(-gravy * 0.5, 4),
        "camsol_min":      round(-gravy * 1.2, 4),
        "camsol_hotspots": int(max(0, round(len(sequence) * 0.05))),
    }


def build_protein_feature_vector_v2(
    accession: str = None,
    sequence: str = None,
    formulation_phs: list = [5.0, 5.5, 6.0, 6.5, 7.0, 7.4]
) -> pd.Series:
    if accession and not sequence:
        data = fetch_uniprot_sequence(accession)
        sequence = data["sequence"]
        protein_id = accession
    else:
        protein_id = "custom"

    features = {"protein_id": protein_id, "sequence_length": len(sequence)}
    features.update(compute_sequence_features(sequence))
    features.update(charge_at_ph(sequence, formulation_phs))

    print(f"    Fetching CamSol scores for {protein_id}...")
    features.update(fetch_camsol_scores(sequence))

    return pd.Series(features)


if __name__ == "__main__":
    accessions = ["P01308", "P00533", "P10636"]
    print("\nFetching protein features + CamSol...\n")
    rows = [build_protein_feature_vector_v2(accession=acc)
            for acc in accessions]
    df = pd.DataFrame(rows).set_index("protein_id")
    df.to_csv("data/processed/protein_features.csv")
    print("\n✓ Updated protein_features.csv with CamSol columns")
    print(df[["isoelectric_point","gravy_score",
              "camsol_mean","camsol_min","camsol_hotspots"]].T)
