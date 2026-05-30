import requests
import pandas as pd
import pubchempy as pcp
import time

# ─────────────────────────────────────────
# 1. Master excipient list with classifications
# Based on approved biologic formulations
# ─────────────────────────────────────────
EXCIPIENT_LIST = [
    # Buffers
    {"name": "histidine",       "class": "buffer",      "mechanism": "pH control, antioxidant"},
    {"name": "citric acid",     "class": "buffer",      "mechanism": "pH control"},
    {"name": "phosphoric acid", "class": "buffer",      "mechanism": "pH control"},
    {"name": "acetic acid",     "class": "buffer",      "mechanism": "pH control"},

    # Sugars / Cryoprotectants
    {"name": "sucrose",         "class": "sugar",       "mechanism": "cryoprotection, stabilization"},
    {"name": "trehalose",       "class": "sugar",       "mechanism": "cryoprotection, stabilization"},
    {"name": "mannitol",        "class": "sugar",       "mechanism": "bulking agent, tonicity"},
    {"name": "sorbitol",        "class": "sugar",       "mechanism": "cryoprotection, tonicity"},

    # Surfactants
    {"name": "polysorbate 80",  "class": "surfactant",  "mechanism": "surface protection, anti-aggregation"},
    {"name": "polysorbate 20",  "class": "surfactant",  "mechanism": "surface protection, anti-aggregation"},
    {"name": "poloxamer 188",   "class": "surfactant",  "mechanism": "surface protection"},

    # Amino acids
    {"name": "arginine",        "class": "amino_acid",  "mechanism": "aggregation suppression, solubility"},
    {"name": "glycine",         "class": "amino_acid",  "mechanism": "stabilization, tonicity"},
    {"name": "proline",         "class": "amino_acid",  "mechanism": "stabilization, cryoprotection"},
    {"name": "methionine",      "class": "amino_acid",  "mechanism": "antioxidant"},

    # Salts
    {"name": "sodium chloride", "class": "salt",        "mechanism": "ionic strength, tonicity"},
    {"name": "potassium chloride","class": "salt",      "mechanism": "ionic strength, tonicity"},

    # Antioxidants
    {"name": "EDTA",            "class": "chelator",    "mechanism": "metal chelation, oxidation prevention"},
    {"name": "ascorbic acid",   "class": "antioxidant", "mechanism": "oxidation prevention"},
]

# ─────────────────────────────────────────
# 2. Fetch physicochemical properties
#    from PubChem for each excipient
# ─────────────────────────────────────────
def fetch_pubchem_properties(excipient_name: str) -> dict:
    try:
        compounds = pcp.get_compounds(excipient_name, 'name')
        if not compounds:
            return {"pubchem_cid": None, "mol_weight": None,
                    "logp": None, "hbd": None, "hba": None,
                    "tpsa": None, "rotatable_bonds": None}

        c = compounds[0]
        return {
            "pubchem_cid":      c.cid,
            "mol_weight":       c.molecular_weight,
            "logp":             c.xlogp,
            "hbd":              c.h_bond_donor_count,
            "hba":              c.h_bond_acceptor_count,
            "tpsa":             c.tpsa,
            "rotatable_bonds":  c.rotatable_bond_count,
        }
    except Exception as e:
        print(f"    Warning: PubChem lookup failed for {excipient_name}: {e}")
        return {"pubchem_cid": None, "mol_weight": None,
                "logp": None, "hbd": None, "hba": None,
                "tpsa": None, "rotatable_bonds": None}

# ─────────────────────────────────────────
# 3. Build full excipient database
# ─────────────────────────────────────────
def build_excipient_database(excipient_list: list) -> pd.DataFrame:
    rows = []
    for exc in excipient_list:
        print(f"  Fetching: {exc['name']}...")
        props = fetch_pubchem_properties(exc['name'])
        row = {**exc, **props}
        rows.append(row)
        time.sleep(0.5)  # be polite to PubChem API

    df = pd.DataFrame(rows)

    # Add concentration range guidance based on approved biologics
    conc_ranges = {
        "histidine":        {"conc_min_mM": 10,  "conc_max_mM": 50},
        "citric acid":      {"conc_min_mM": 10,  "conc_max_mM": 50},
        "phosphoric acid":  {"conc_min_mM": 10,  "conc_max_mM": 50},
        "acetic acid":      {"conc_min_mM": 10,  "conc_max_mM": 50},
        "sucrose":          {"conc_min_mM": 100, "conc_max_mM": 300},
        "trehalose":        {"conc_min_mM": 100, "conc_max_mM": 300},
        "mannitol":         {"conc_min_mM": 100, "conc_max_mM": 300},
        "sorbitol":         {"conc_min_mM": 100, "conc_max_mM": 300},
        "polysorbate 80":   {"conc_min_mM": 0.01,"conc_max_mM": 0.08},
        "polysorbate 20":   {"conc_min_mM": 0.01,"conc_max_mM": 0.08},
        "poloxamer 188":    {"conc_min_mM": 0.01,"conc_max_mM": 0.05},
        "arginine":         {"conc_min_mM": 50,  "conc_max_mM": 150},
        "glycine":          {"conc_min_mM": 50,  "conc_max_mM": 200},
        "proline":          {"conc_min_mM": 50,  "conc_max_mM": 150},
        "methionine":       {"conc_min_mM": 1,   "conc_max_mM": 10},
        "sodium chloride":  {"conc_min_mM": 50,  "conc_max_mM": 150},
        "potassium chloride":{"conc_min_mM": 10, "conc_max_mM": 50},
        "EDTA":             {"conc_min_mM": 0.05,"conc_max_mM": 0.1},
        "ascorbic acid":    {"conc_min_mM": 1,   "conc_max_mM": 5},
    }

    df["conc_min_mM"] = df["name"].map(
        lambda x: conc_ranges.get(x, {}).get("conc_min_mM"))
    df["conc_max_mM"] = df["name"].map(
        lambda x: conc_ranges.get(x, {}).get("conc_max_mM"))

    return df

# ─────────────────────────────────────────
# 4. Run
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\nBuilding excipient database from PubChem...\n")
    df = build_excipient_database(EXCIPIENT_LIST)

    df.to_csv("data/processed/excipient_db.csv", index=False)

    print(f"\n✓ Done! Excipient database shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nPreview:")
    print(df[["name","class","mol_weight","logp","conc_min_mM","conc_max_mM"]].to_string(index=False))
    print("\nSaved to: data/processed/excipient_db.csv")
