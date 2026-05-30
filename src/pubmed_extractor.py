import requests
import pandas as pd
import time
import re

# ─────────────────────────────────────────
# 1. PubMed E-utilities API
#    No API key needed for small volumes
#    With key: 10 req/sec, Without: 3 req/sec
# ─────────────────────────────────────────
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def search_pubmed(query: str, retmax: int = 20) -> list:
    """
    Search PubMed and return list of PMIDs.
    """
    url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  retmax,
        "retmode": "json",
        "sort":    "relevance",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        pmids = data["esearchresult"]["idlist"]
        return pmids
    except Exception as e:
        print(f"    Search error: {e}")
        return []

def fetch_abstracts(pmids: list) -> list:
    """
    Fetch abstract text for a list of PMIDs.
    Returns list of dicts with pmid, title, abstract.
    """
    if not pmids:
        return []

    url = f"{EUTILS_BASE}/efetch.fcgi"
    params = {
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        xml  = resp.text

        # Extract articles using regex on XML
        articles = []
        # Split by article
        article_blocks = re.findall(
            r'<PubmedArticle>(.*?)</PubmedArticle>',
            xml, re.DOTALL
        )

        for block in article_blocks:
            # PMID
            pmid_match = re.search(r'<PMID[^>]*>(\d+)</PMID>', block)
            pmid = pmid_match.group(1) if pmid_match else ""

            # Title
            title_match = re.search(
                r'<ArticleTitle>(.*?)</ArticleTitle>', block, re.DOTALL)
            title = re.sub(r'<[^>]+>', '', title_match.group(1)) \
                    if title_match else ""

            # Abstract
            abs_match = re.search(
                r'<AbstractText[^>]*>(.*?)</AbstractText>',
                block, re.DOTALL)
            abstract = re.sub(r'<[^>]+>', '', abs_match.group(1)) \
                       if abs_match else ""

            # Journal
            journal_match = re.search(
                r'<Title>(.*?)</Title>', block, re.DOTALL)
            journal = re.sub(r'<[^>]+>', '', journal_match.group(1)) \
                      if journal_match else ""

            # Year
            year_match = re.search(r'<Year>(\d{4})</Year>', block)
            year = year_match.group(1) if year_match else ""

            if pmid and (title or abstract):
                articles.append({
                    "pmid":     pmid,
                    "title":    title.strip(),
                    "abstract": abstract.strip(),
                    "journal":  journal.strip(),
                    "year":     year,
                })
        return articles

    except Exception as e:
        print(f"    Fetch error: {e}")
        return []

# ─────────────────────────────────────────
# 2. Parse formulation evidence from text
# ─────────────────────────────────────────
EXCIPIENT_PATTERNS = {
    "histidine":          r"histidine",
    "citric_acid":        r"citric acid",
    "phosphate":          r"phosphat",
    "acetate":            r"sodium acetate|acetic acid",
    "succinate":          r"succinate",
    "sucrose":            r"sucrose",
    "trehalose":          r"trehalose",
    "mannitol":           r"mannitol",
    "sorbitol":           r"sorbitol",
    "polysorbate_80":     r"polysorbate.?80|tween.?80|ps.?80",
    "polysorbate_20":     r"polysorbate.?20|tween.?20|ps.?20",
    "poloxamer_188":      r"poloxamer.?188|pluronic",
    "arginine":           r"arginine",
    "glycine":            r"glycine",
    "proline":            r"proline",
    "methionine":         r"methionine",
    "sodium_chloride":    r"sodium chloride|nacl",
    "edta":               r"edta|edetate",
    "ascorbic_acid":      r"ascorbic acid",
}

STABILITY_OUTCOMES = {
    "aggregation":   r"aggregat|particle|turbidity|opalescen",
    "oxidation":     r"oxidat|methionine oxidat|tryptophan oxidat",
    "deamidation":   r"deamidat",
    "potency":       r"potency|bioactivity|binding activity",
    "viscosity":     r"viscosity|viscous",
    "shelf_life":    r"shelf.?life|stability study|long.?term stab",
    "lyophilized":   r"lyophiliz|freeze.?dr",
    "liquid":        r"liquid formul|solution formul",
}

PROTEIN_TYPES = {
    "monoclonal_antibody": r"monoclonal antibody|mab|igg|antibody",
    "bispecific":          r"bispecific",
    "fusion_protein":      r"fusion protein|fc.fusion",
    "insulin":             r"insulin",
    "growth_factor":       r"growth factor|egf|vegf|erythropoiet",
    "enzyme":              r"enzyme|lipase|amylase|glucocerebrosidase",
    "vaccine":             r"vaccine|antigen|adjuvant",
}

def parse_abstract(text: str) -> dict:
    """Extract formulation evidence from abstract text."""
    text_lower = text.lower()
    result = {}

    # Excipients mentioned
    for exc, pattern in EXCIPIENT_PATTERNS.items():
        result[f"mentions_{exc}"] = int(
            bool(re.search(pattern, text_lower)))

    # Stability outcomes studied
    for outcome, pattern in STABILITY_OUTCOMES.items():
        result[f"studies_{outcome}"] = int(
            bool(re.search(pattern, text_lower)))

    # Protein type
    for ptype, pattern in PROTEIN_TYPES.items():
        result[f"protein_{ptype}"] = int(
            bool(re.search(pattern, text_lower)))

    # pH mentioned
    ph_match = re.search(
        r"ph\s*(?:of\s*)?(\d+\.?\d*)\s*(?:to|-)\s*(\d+\.?\d*)",
        text_lower)
    if ph_match:
        result["ph_mentioned"] = round(
            (float(ph_match.group(1)) +
             float(ph_match.group(2))) / 2, 1)
    else:
        ph_match2 = re.search(
            r"ph\s*(?:of\s*|:?\s*)(\d+\.?\d*)", text_lower)
        if ph_match2:
            ph = float(ph_match2.group(1))
            result["ph_mentioned"] = ph if 3.0 <= ph <= 9.0 else None
        else:
            result["ph_mentioned"] = None

    # Stability outcome sentiment (positive/negative)
    positive_terms = r"improv|enhanc|stabiliz|protect|reduc.*aggregat|prevent"
    negative_terms = r"degrad|instab|aggregat.*increas|loss of potency"
    result["positive_outcome"] = int(
        bool(re.search(positive_terms, text_lower)))
    result["negative_outcome"] = int(
        bool(re.search(negative_terms, text_lower)))

    return result

# ─────────────────────────────────────────
# 3. Master pipeline
# ─────────────────────────────────────────
def build_pubmed_dataset(queries: list,
                          retmax_per_query: int = 25) -> pd.DataFrame:
    all_articles = []
    seen_pmids   = set()

    for query in queries:
        print(f"\n  Query: '{query}'")
        pmids = search_pubmed(query, retmax=retmax_per_query)
        print(f"  Found {len(pmids)} PMIDs")

        # Filter already seen
        new_pmids = [p for p in pmids if p not in seen_pmids]
        seen_pmids.update(new_pmids)

        if not new_pmids:
            continue

        articles = fetch_abstracts(new_pmids)
        print(f"  Fetched {len(articles)} abstracts")

        for art in articles:
            full_text = art["title"] + " " + art["abstract"]
            parsed    = parse_abstract(full_text)
            row = {
                "source":   "PubMed",
                "pmid":     art["pmid"],
                "title":    art["title"],
                "journal":  art["journal"],
                "year":     art["year"],
                "query":    query,
                **parsed
            }
            all_articles.append(row)

        time.sleep(0.4)

    return pd.DataFrame(all_articles)

# ─────────────────────────────────────────
# 4. Run
# ─────────────────────────────────────────
if __name__ == "__main__":
    queries = [
        "protein formulation stability excipient",
        "monoclonal antibody formulation aggregation",
        "biopharmaceutical excipient selection stability",
        "protein aggregation prevention surfactant",
        "antibody formulation polysorbate stability",
        "protein lyophilization sucrose trehalose",
        "biologic drug formulation pH stability",
        "protein oxidation methionine formulation",
        "monoclonal antibody histidine buffer stability",
        "protein deamidation asparagine formulation",
    ]

    print("\nExtracting PubMed formulation literature...\n")
    df = build_pubmed_dataset(queries, retmax_per_query=25)

    df.to_csv("data/raw/pubmed_formulation_literature.csv", index=False)

    print(f"\n{'='*55}")
    print(f"✓ Extracted {len(df)} paper abstracts")
    print(f"  Unique PMIDs: {df['pmid'].nunique()}")

    # Excipient mentions
    mention_cols = [c for c in df.columns if c.startswith("mentions_")]
    print(f"\nExcipient mention frequency in literature:")
    freq = df[mention_cols].mean().sort_values(ascending=False)
    for col, val in freq.items():
        if val > 0:
            bar = "█" * int(val * 30)
            exc = col.replace("mentions_","")
            print(f"  {exc:<25} {bar:<30} {val:.0%}")

    # Outcomes studied
    outcome_cols = [c for c in df.columns if c.startswith("studies_")]
    print(f"\nStability outcomes studied:")
    ofreq = df[outcome_cols].mean().sort_values(ascending=False)
    for col, val in ofreq.items():
        if val > 0:
            bar = "█" * int(val * 30)
            out = col.replace("studies_","")
            print(f"  {out:<25} {bar:<30} {val:.0%}")

    print(f"\nProtein types covered:")
    prot_cols = [c for c in df.columns if c.startswith("protein_")]
    pfreq = df[prot_cols].mean().sort_values(ascending=False)
    for col, val in pfreq.items():
        if val > 0:
            bar = "█" * int(val * 30)
            ptype = col.replace("protein_","")
            print(f"  {ptype:<25} {bar:<30} {val:.0%}")

    print(f"\nSaved to: data/raw/pubmed_formulation_literature.csv")
