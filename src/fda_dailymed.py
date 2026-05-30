import requests
import pandas as pd
import time
import re

BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

# ─────────────────────────────────────────
# 1. Search DailyMed
# ─────────────────────────────────────────
def search_biologics(keywords: list, pagesize: int = 10) -> list:
    results = []
    for keyword in keywords:
        print(f"  Searching: '{keyword}'...")
        url = f"{BASE_URL}/spls.json"
        params = {"drug_name": keyword, "pagesize": pagesize}
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            spls = data.get("data", [])
            for spl in spls:
                results.append({
                    "keyword": keyword,
                    "setid":   spl.get("setid", ""),
                    "title":   spl.get("title", ""),
                })
        except Exception as e:
            print(f"    Warning: {e}")
        time.sleep(0.3)
    return results

# ─────────────────────────────────────────
# 2. Fetch full label text via NDC endpoint
# ─────────────────────────────────────────
def fetch_label_text(setid: str) -> str:
    """
    Fetches the complete SPL XML label and extracts
    all text content for parsing.
    """
    # Try the applicationxml endpoint first
    url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            # Strip XML tags, return raw text
            text = re.sub(r'<[^>]+>', ' ', resp.text)
            text = re.sub(r'\s+', ' ', text)
            return text.lower()
    except Exception:
        pass

    # Fallback: use JSON endpoint sections
    url2 = f"{BASE_URL}/spls/{setid}.json"
    try:
        resp2 = requests.get(url2, timeout=15)
        resp2.raise_for_status()
        data = resp2.json().get("data", {})

        # Concatenate all section text
        full_text = ""
        full_text += str(data.get("title", "")) + " "
        full_text += str(data.get("dosage_form", "")) + " "
        full_text += str(data.get("route", "")) + " "

        sections = data.get("sections", [])
        for sec in sections:
            full_text += str(sec.get("name", "")) + " "
            full_text += str(sec.get("text", "")) + " "
            # Some APIs nest content differently
            content = sec.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        full_text += str(item.get("text","")) + " "
                    else:
                        full_text += str(item) + " "

        return full_text.lower()
    except Exception as e:
        return ""

# ─────────────────────────────────────────
# 3. Also try the human readable label URL
# ─────────────────────────────────────────
def fetch_label_html(setid: str) -> str:
    """Fetch human-readable label HTML as fallback."""
    url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            text = re.sub(r'<[^>]+>', ' ', resp.text)
            text = re.sub(r'\s+', ' ', text)
            return text.lower()
    except Exception:
        pass
    return ""

# ─────────────────────────────────────────
# 4. Parse excipients from text
# ─────────────────────────────────────────
EXCIPIENT_PATTERNS = {
    "histidine":          r"histidine",
    "citric_acid":        r"citric acid",
    "phosphate":          r"phosphat",
    "acetate":            r"acetic acid|sodium acetate",
    "succinate":          r"succinic acid|sodium succinate",
    "sucrose":            r"sucrose",
    "trehalose":          r"trehalose",
    "mannitol":           r"mannitol",
    "sorbitol":           r"sorbitol",
    "polysorbate_80":     r"polysorbate 80|tween 80",
    "polysorbate_20":     r"polysorbate 20|tween 20",
    "poloxamer_188":      r"poloxamer 188|pluronic",
    "arginine":           r"arginine",
    "glycine":            r"glycine",
    "proline":            r"proline",
    "methionine":         r"methionine",
    "sodium_chloride":    r"sodium chloride|nacl",
    "potassium_chloride": r"potassium chloride",
    "edta":               r"edta|edetate",
    "ascorbic_acid":      r"ascorbic acid",
}

def parse_excipients(text: str) -> dict:
    flags = {}
    for exc, pattern in EXCIPIENT_PATTERNS.items():
        flags[f"has_{exc}"] = int(bool(re.search(pattern, text)))
    return flags

def extract_ph(text: str) -> float:
    match = re.search(r"ph\s*(?:of\s*|range\s*)?(\d+\.?\d*)\s*(?:to|-)\s*(\d+\.?\d*)", text)
    if match:
        return round((float(match.group(1)) + float(match.group(2))) / 2, 1)
    match2 = re.search(r"ph\s*(?:of\s*|:?\s*)(\d+\.?\d*)", text)
    if match2:
        ph = float(match2.group(1))
        if 3.0 <= ph <= 9.0:
            return ph
    return None

def extract_storage(text: str) -> str:
    if re.search(r"2\s*[°o]?\s*c.*8\s*[°o]?\s*c|refrigerat", text):
        return "2-8C"
    elif re.search(r"room temp|20\s*[°o]?\s*c.*25|25\s*[°o]?\s*c", text):
        return "25C"
    elif re.search(r"frozen|\-20|\-80", text):
        return "frozen"
    return "unknown"

# ─────────────────────────────────────────
# 5. Master pipeline
# ─────────────────────────────────────────
def build_dailymed_dataset(keywords: list) -> pd.DataFrame:
    print("\nSearching DailyMed...")
    spls = search_biologics(keywords)
    print(f"  Total labels found: {len(spls)}")

    rows = []
    seen = set()

    for spl in spls:
        setid = spl["setid"]
        if setid in seen or not setid:
            continue
        seen.add(setid)

        print(f"  Processing: {spl['title'][:55]}...")

        # Try XML first, fallback to JSON, fallback to HTML
        text = fetch_label_text(setid)
        if len(text) < 200:
            print(f"    Trying HTML fallback...")
            text = fetch_label_html(setid)

        exc_flags = parse_excipients(text)
        ph        = extract_ph(text)
        storage   = extract_storage(text)

        # Check if we found anything
        found = sum(exc_flags.values())
        print(f"    Excipients found: {found} | pH: {ph} | Storage: {storage}")

        row = {
            "source":      "FDA_DailyMed",
            "drug_name":   spl["title"],
            "keyword":     spl["keyword"],
            "ph":          ph,
            "storage":     storage,
            **exc_flags
        }
        rows.append(row)
        time.sleep(0.4)

    return pd.DataFrame(rows)

# ─────────────────────────────────────────
# 6. Run
# ─────────────────────────────────────────
if __name__ == "__main__":
    keywords = [
        "adalimumab", "trastuzumab", "bevacizumab",
        "rituximab",  "infliximab",  "insulin",
        "etanercept", "pembrolizumab","nivolumab",
        "durvalumab",
    ]

    df = build_dailymed_dataset(keywords)

    df.to_csv("data/raw/fda_dailymed_formulations.csv", index=False)

    print(f"\n{'='*55}")
    print(f"✓ Extracted {len(df)} biologic formulations")

    exc_cols = [c for c in df.columns if c.startswith("has_")]
    prev = df[exc_cols].mean().sort_values(ascending=False)

    print(f"\nExcipient prevalence in approved biologics:")
    for col, val in prev.items():
        if val > 0:
            bar = "█" * int(val * 30)
            print(f"  {col:<25} {bar:<30} {val:.0%}")

    print(f"\nFormulations with at least 1 excipient detected:")
    print(f"  {(df[exc_cols].sum(axis=1) > 0).sum()} / {len(df)}")
    print(f"\nSaved to: data/raw/fda_dailymed_formulations.csv")
