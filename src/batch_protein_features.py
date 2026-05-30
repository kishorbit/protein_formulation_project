import pandas as pd
import numpy as np
import requests
import time
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import sys
sys.path.insert(0, "src")
from protein_features import (compute_sequence_features,
                               charge_at_ph,
                               fetch_camsol_scores)

def process_one_protein(row: pd.Series,
                        formulation_phs: list) -> dict:
    acc      = row["accession"]
    sequence = row["sequence"]
    if not sequence or len(sequence) < 50:
        return None

    features = {
        "protein_id":    acc,
        "protein_name":  row.get("protein_name",""),
        "gene":          row.get("gene",""),
        "query_label":   row.get("query_label",""),
        "sequence_length": len(sequence),
    }

    try:
        features.update(compute_sequence_features(sequence))
    except Exception as e:
        print(f"    ProtParam failed for {acc}: {e}")
        return None

    try:
        features.update(charge_at_ph(sequence, formulation_phs))
    except Exception as e:
        print(f"    Charge calc failed for {acc}: {e}")

    try:
        features.update(fetch_camsol_scores(sequence))
    except Exception as e:
        print(f"    CamSol failed for {acc}: {e}")
        features.update({
            "camsol_mean": np.nan,
            "camsol_min":  np.nan,
            "camsol_hotspots": np.nan,
        })

    return features


if __name__ == "__main__":
    protein_list = pd.read_csv("data/raw/protein_list.csv")
    print(f"\nProcessing {len(protein_list)} proteins...\n")

    formulation_phs = [5.0, 5.5, 6.0, 6.5, 7.0, 7.4]
    rows = []
    failed = []

    for i, row in protein_list.iterrows():
        acc = row["accession"]
        print(f"  [{i+1}/{len(protein_list)}] {acc} "
              f"({row.get('protein_name','')[:40]})")

        result = process_one_protein(row, formulation_phs)
        if result:
            rows.append(result)
            print(f"    ✓ {len(result)} features extracted")
        else:
            failed.append(acc)
            print(f"    ✗ Skipped")

        time.sleep(0.2)

    df = pd.DataFrame(rows)
    df.to_csv("data/processed/protein_features.csv", index=False)

    print(f"\n{'='*55}")
    print(f"✓ Processed: {len(rows)} proteins")
    print(f"✗ Failed:    {len(failed)} proteins")
    if failed:
        print(f"  Failed accessions: {failed}")
    print(f"\nFeature matrix shape: {df.shape}")
    print(f"\nKey features summary:")
    key_cols = ["isoelectric_point","gravy_score","instability_index",
                "molecular_weight_kda","camsol_mean","camsol_hotspots"]
    available = [c for c in key_cols if c in df.columns]
    print(df[["protein_id"] + available].to_string(index=False))
    print(f"\nSaved to: data/processed/protein_features.csv")
