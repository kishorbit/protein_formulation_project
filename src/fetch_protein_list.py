import requests
import pandas as pd
import time

def fetch_therapeutic_proteins(max_results: int = 50) -> pd.DataFrame:
    """
    Query UniProt for reviewed human therapeutic proteins
    used in biopharmaceuticals — monoclonal antibodies,
    insulins, growth factors, enzymes, cytokines.
    """
    url = "https://rest.uniprot.org/uniprotkb/search"

    # Reviewed human proteins with pharmaceutical annotation
    queries = [
        # Monoclonal antibodies / immunoglobulins
        ("monoclonal antibody human",
         "reviewed:true AND organism_id:9606 AND "
         "keyword:KW-0472 AND length:[100 TO 1500]"),

        # Insulins and related
        ("insulin human",
         "reviewed:true AND organism_id:9606 AND "
         "name:insulin AND length:[50 TO 300]"),

        # Growth factors
        ("growth factor therapeutic human",
         "reviewed:true AND organism_id:9606 AND "
         "keyword:KW-0339 AND length:[100 TO 800]"),

        # Enzymes therapeutic
        ("therapeutic enzyme human",
         "reviewed:true AND organism_id:9606 AND "
         "keyword:KW-0240 AND keyword:KW-0488 AND "
         "length:[200 TO 1000]"),

        # Cytokines / interleukins
        ("cytokine interleukin human therapeutic",
         "reviewed:true AND organism_id:9606 AND "
         "keyword:KW-0202 AND length:[100 TO 600]"),

        # Interferons
        ("interferon human",
         "reviewed:true AND organism_id:9606 AND "
         "name:interferon AND length:[100 TO 600]"),

        # Erythropoietin / colony stimulating factors
        ("erythropoietin colony stimulating human",
         "reviewed:true AND organism_id:9606 AND "
         "keyword:KW-0339 AND name:erythropoietin OR "
         "name:thrombopoietin AND length:[150 TO 600]"),
    ]

    all_rows = []
    seen_accessions = set()

    for label, query in queries:
        print(f"  Querying: {label}...")
        params = {
            "query":   query,
            "format":  "json",
            "size":    min(max_results // len(queries) + 5, 25),
            "fields":  "accession,id,protein_name,gene_names,length,sequence",
        }
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            print(f"    Got {len(results)} entries")

            for entry in results:
                acc = entry.get("primaryAccession","")
                if acc in seen_accessions or not acc:
                    continue
                seen_accessions.add(acc)

                # Extract protein name
                pname = ""
                pnames = entry.get("proteinDescription",{})
                rec = pnames.get("recommendedName",{})
                if rec:
                    full = rec.get("fullName",{})
                    pname = full.get("value","")
                if not pname:
                    sub = pnames.get("submissionNames",[])
                    if sub:
                        pname = sub[0].get("fullName",{}).get("value","")

                # Extract gene name
                genes = entry.get("genes",[])
                gene  = genes[0].get("geneName",{}).get("value","") \
                        if genes else ""

                # Extract sequence
                seq_data = entry.get("sequence",{})
                seq      = seq_data.get("value","")
                length   = seq_data.get("length", len(seq))

                all_rows.append({
                    "accession":    acc,
                    "protein_name": pname,
                    "gene":         gene,
                    "length":       length,
                    "sequence":     seq,
                    "query_label":  label,
                })
        except Exception as e:
            print(f"    Warning: {e}")
        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    # Remove very short sequences (fragments) and very long (multimers)
    df = df[(df["length"] >= 80) & (df["length"] <= 1400)]
    df = df.drop_duplicates("accession").reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("\nFetching therapeutic protein list from UniProt...\n")
    df = fetch_therapeutic_proteins(max_results=60)

    df.to_csv("data/raw/protein_list.csv", index=False)

    print(f"\n{'='*55}")
    print(f"✓ Retrieved {len(df)} unique therapeutic proteins")
    print(f"\nBreakdown by query type:")
    print(df["query_label"].value_counts().to_string())
    print(f"\nLength distribution:")
    print(df["length"].describe().round(0).to_string())
    print(f"\nProtein list preview:")
    print(df[["accession","protein_name","gene","length"]
             ].to_string(index=False))
    print(f"\nSaved to: data/raw/protein_list.csv")
