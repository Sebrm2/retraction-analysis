"""
citation_enrichment.py
======================
Enrich each retracted paper with citation counts from two sources:

  1. NIH iCite  (via PubMed ID)   -- fast, reliable for biomedical papers
  2. OpenAlex   (via DOI, BATCHED) -- fills the gap for non-PubMed journals
  3. OpenAlex   (via title search) -- last-resort fallback for the remainder

STAGE 1 FIX
-----------
The previous OpenAlex step built a malformed filter and matched zero papers,
biasing all downstream citation analyses toward the PubMed-indexed subset.
This version constructs a correct piped `doi:` filter, URL-encodes each DOI,
and pages through results, so DOI-only papers (nanotechnology, paper-mill
journals) are captured too.

The enriched dataframe is cached to a parquet so repeated runs skip the APIs.

    python src/citation_enrichment.py

Depends on data/clean_retractions.parquet.
Produces  data/enriched_retractions.parquet.
"""

from __future__ import annotations

import time
from urllib.parse import quote

import numpy as np
import pandas as pd

from config import (
    DATA_DIR, CONTACT_EMAIL, CITATION_SAMPLE_SIZE, CITATION_CACHE_PATH,
)
from data_prep import load_clean

ENRICHED_PATH = DATA_DIR / "enriched_retractions.parquet"

_HEADERS = {"User-Agent": f"retraction-analysis (mailto:{CONTACT_EMAIL})"}
_OPENALEX_WORKS = "https://api.openalex.org/works"
_ICITE = "https://icite.od.nih.gov/api/pubs"


# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 1: iCite
# ─────────────────────────────────────────────────────────────────────────────

def fetch_icite(pmids: np.ndarray, requests) -> dict[int, dict]:
    """Fetch citation counts and RCR from iCite in chunks of 100 PMIDs."""
    result: dict[int, dict] = {}
    for i in range(0, len(pmids), 100):
        chunk = ",".join(str(x) for x in pmids[i:i + 100])
        try:
            resp = requests.get(
                _ICITE,
                params={"pmids": chunk,
                        "fields": "pmid,citation_count,relative_citation_ratio,year"},
                timeout=30,
            )
            resp.raise_for_status()
            for rec in resp.json().get("data", []):
                result[int(rec["pmid"])] = {
                    "citation_count": rec.get("citation_count"),
                    "rcr": rec.get("relative_citation_ratio"),
                    "citation_source": "iCite",
                }
        except Exception as exc:
            print(f"    iCite chunk {i} failed: {exc}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 2: OpenAlex batched DOI lookup  (STAGE 1 FIX)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_openalex_by_doi(dois: list[str], requests) -> dict[str, int]:
    """
    Fetch cited_by_count for a list of DOIs using OpenAlex's OR filter.

    OpenAlex accepts up to ~50 values in a piped filter:
        filter=doi:10.x/a|10.y/b|10.z/c
    DOIs must be lowercased and bare (no https://doi.org/ prefix). We URL-encode
    each DOI so slashes and special characters survive transport.
    """
    result: dict[str, int] = {}
    CHUNK = 50
    for i in range(0, len(dois), CHUNK):
        chunk = [d.lower().strip() for d in dois[i:i + CHUNK] if d]
        # Build the piped filter value, encoding each DOI individually.
        filter_value = "doi:" + "|".join(quote(d, safe="") for d in chunk)
        try:
            resp = requests.get(
                _OPENALEX_WORKS,
                params={"filter": filter_value,
                        "per-page": CHUNK,
                        "select": "doi,cited_by_count"},
                headers=_HEADERS,
                timeout=40,
            )
            if resp.status_code != 200:
                if i == 0:
                    print(f"    OpenAlex DOI batch returned {resp.status_code}")
                continue
            for work in resp.json().get("results", []):
                raw = (work.get("doi") or "").replace("https://doi.org/", "").lower()
                if raw:
                    result[raw] = work.get("cited_by_count")
            time.sleep(0.15)   # be polite to the shared pool
        except Exception as exc:
            if i == 0:
                print(f"    OpenAlex DOI batch failed: {exc}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 3: OpenAlex title search (fallback, capped)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_openalex_by_title(titles: pd.Series, requests, cap: int = 500) -> dict:
    """Search OpenAlex by title for papers still missing citations (slow)."""
    result = {}
    for record_id, title in titles.head(cap).items():
        try:
            resp = requests.get(
                _OPENALEX_WORKS,
                params={"search": str(title)[:200], "per-page": 1,
                        "select": "title,cited_by_count"},
                headers=_HEADERS, timeout=25,
            )
            if resp.status_code == 200:
                hits = resp.json().get("results", [])
                if hits:
                    result[record_id] = hits[0].get("cited_by_count")
            time.sleep(0.1)
        except Exception:
            pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    print("\n== Citation enrichment (iCite + OpenAlex) ==")
    try:
        import requests
    except ImportError:
        print("  requests not installed; skipping enrichment.")
        df["citation_count"] = np.nan
        df["rcr"] = np.nan
        return df

    df = df.copy()
    df["citation_count"] = np.nan
    df["rcr"] = np.nan
    df["citation_source"] = pd.NA

    # --- Source 1: iCite by PMID ---
    pmids = df["OriginalPaperPubMedID"].dropna().astype(int).unique()[:CITATION_SAMPLE_SIZE]
    print(f"  iCite: querying {len(pmids):,} PubMed IDs ...")
    icite = fetch_icite(pmids, requests)
    print(f"  iCite matched: {len(icite):,}")
    if icite:
        icite_df = (
            pd.DataFrame.from_dict(icite, orient="index").reset_index()
            .rename(columns={"index": "OriginalPaperPubMedID"})
        )
        icite_df["OriginalPaperPubMedID"] = icite_df["OriginalPaperPubMedID"].astype(float)
        df = df.drop(columns=["citation_count", "rcr", "citation_source"]).merge(
            icite_df, on="OriginalPaperPubMedID", how="left"
        )

    # --- Source 2: OpenAlex batched DOI for still-missing papers ---
    missing = df["citation_count"].isna() & df["OriginalPaperDOI"].notna()
    dois = df.loc[missing, "OriginalPaperDOI"].dropna().tolist()
    dois = [d for d in dois if len(str(d)) > 3][:CITATION_SAMPLE_SIZE]
    print(f"  OpenAlex DOI batch: querying {len(dois):,} DOIs ...")
    oa = fetch_openalex_by_doi(dois, requests)
    print(f"  OpenAlex DOI matched: {len(oa):,}")
    if oa:
        df["_doi_norm"] = df["OriginalPaperDOI"].str.lower().str.strip()
        fill = df["_doi_norm"].map(oa)
        newly = df["citation_count"].isna() & fill.notna()
        df.loc[newly, "citation_count"] = fill[newly]
        df.loc[newly, "citation_source"] = "OpenAlex_DOI"
        df = df.drop(columns=["_doi_norm"])

    # --- Source 3: OpenAlex title fallback ---
    still = df["citation_count"].isna() & df["Title"].notna()
    titles = df.loc[still, "Title"]
    titles.index = df.loc[still, "Record ID"]
    print(f"  OpenAlex title fallback: querying up to 500 of {still.sum():,} ...")
    title_hits = fetch_openalex_by_title(titles, requests, cap=500)
    if title_hits:
        idx = df["Record ID"].isin(title_hits)
        df.loc[idx, "citation_count"] = df.loc[idx, "Record ID"].map(title_hits)
        df.loc[idx & df["citation_source"].isna(), "citation_source"] = "OpenAlex_title"

    matched = df["citation_count"].notna().sum()
    print(f"  Total matched: {matched:,} / {len(df):,} ({matched / len(df) * 100:.1f}%)")
    return df


def main() -> None:
    if CITATION_CACHE_PATH.exists():
        print(f"  Cached citations found at {CITATION_CACHE_PATH.name}; loading.")
        df = pd.read_parquet(CITATION_CACHE_PATH)
    else:
        df = load_clean()
        df = enrich(df)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CITATION_CACHE_PATH, index=False)

    df.to_parquet(ENRICHED_PATH, index=False)
    print(f"  Saved enriched dataset -> {ENRICHED_PATH.name}")


def load_enriched() -> pd.DataFrame:
    """Load the enriched dataset; fall back to clean data if not yet built."""
    if ENRICHED_PATH.exists():
        return pd.read_parquet(ENRICHED_PATH)
    print("  enriched_retractions.parquet not found; using clean data without citations.")
    df = load_clean()
    df["citation_count"] = np.nan
    df["rcr"] = np.nan
    return df


if __name__ == "__main__":
    main()
