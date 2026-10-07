"""
data_prep.py
============
Stage 0 of the pipeline: load the raw Retraction Watch CSV, filter to the
study's biomedical subject scope, clean the fields, and engineer the derived
features every downstream stage relies on.

Run directly to produce a cleaned parquet that the other stages load:

    python src/data_prep.py

Output: data/clean_retractions.parquet
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from config import (
    DATA_PATH, DATA_DIR, EXPLORE_SUBJECTS, BIOMEDICAL_SUBJECTS,
    SUBJECT_DISPLAY_NAMES, RETRACTION_NATURE, REASON_CATEGORIES,
)

CLEAN_PATH = DATA_DIR / "clean_retractions.parquet"


# ─────────────────────────────────────────────────────────────────────────────
#  REASON MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# Precompute a lowercase lookup: raw reason -> category label.
_REASON_LOOKUP = {
    member.lower(): category
    for category, members in REASON_CATEGORIES.items()
    for member in members
}


def map_reasons(raw_str) -> list[str]:
    """Convert a semicolon-separated raw reason string to category labels."""
    if pd.isna(raw_str):
        return ["Other / unknown"]
    categories = set()
    for token in str(raw_str).split(";"):
        token = token.strip()
        if not token:
            continue
        categories.add(_REASON_LOOKUP.get(token.lower(), "Other / unknown"))
    return list(categories) if categories else ["Other / unknown"]


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD & FILTER
# ─────────────────────────────────────────────────────────────────────────────

def load_and_filter() -> pd.DataFrame | None:
    """Load the raw CSV, keep true retractions, filter to the subject scope."""
    print("== Data ingestion & subject filter ==")
    if not DATA_PATH.exists():
        sys.exit(
            f"ERROR: {DATA_PATH} not found.\n"
            "Download the Retraction Watch CSV and place it in data/."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  Total records                 : {len(df):>8,}")

    df = df[df["RetractionNature"].str.strip() == RETRACTION_NATURE].copy()
    print(f"  After RetractionNature filter : {len(df):>8,}")

    if EXPLORE_SUBJECTS:
        subjects = (
            df["Subject"].dropna().str.split(";").explode().str.strip().unique()
        )
        print("\n  Unique Subject tags in the retraction-only subset:\n")
        for tag in sorted(subjects):
            print(f"    {tag}")
        print(
            "\n  Paste the tags you want into BIOMEDICAL_SUBJECTS in config.py, "
            "set EXPLORE_SUBJECTS=False, and re-run."
        )
        return None

    if not BIOMEDICAL_SUBJECTS:
        sys.exit("BIOMEDICAL_SUBJECTS is empty in config.py.")

    pattern = "|".join(re.escape(s) for s in BIOMEDICAL_SUBJECTS)
    mask = df["Subject"].fillna("").str.contains(pattern, case=False, regex=True)
    df = df[mask].copy()
    print(f"  After biomedical filter       : {len(df):>8,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  CLEAN & FEATURE ENGINEER
# ─────────────────────────────────────────────────────────────────────────────

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, normalise identifiers, and derive analysis features."""
    print("\n== Cleaning & feature engineering ==")

    # Dates and time-to-retraction
    df["OriginalPaperDate"] = pd.to_datetime(df["OriginalPaperDate"], errors="coerce")
    df["RetractionDate"]    = pd.to_datetime(df["RetractionDate"], errors="coerce")
    df["pub_year"]          = df["OriginalPaperDate"].dt.year.astype("Int64")
    df["retraction_year"]   = df["RetractionDate"].dt.year.astype("Int64")
    df["time_to_retraction_days"]  = (df["RetractionDate"] - df["OriginalPaperDate"]).dt.days
    df["time_to_retraction_years"] = df["time_to_retraction_days"] / 365.25

    # DOIs and PubMed IDs: normalise missing sentinels to NaN
    for col in ("RetractionDOI", "OriginalPaperDOI"):
        df[col] = df[col].fillna("").str.strip()
        df[col] = df[col].where(~df[col].isin({"unavailable", "Unavailable", ""}), other=np.nan)
    for col in ("RetractionPubMedID", "OriginalPaperPubMedID"):
        df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)

    df["has_retraction_doi"] = df["RetractionDOI"].notna()
    df["paywalled"] = (
        df["Paywalled"].fillna("").str.strip().str.upper().map({"YES": True, "NO": False})
    )

    # Reason categories
    df["ReasonCategories"] = df["Reason"].apply(map_reasons)

    # Author count
    df["author_count"] = (
        df["Author"].fillna("").str.split(";")
        .apply(lambda names: len([a for a in names if a.strip()]))
        .replace(0, np.nan)
    )

    # Publisher / decade flags
    df["is_hindawi"]  = df["Publisher"].fillna("").str.contains("Hindawi", case=False)
    df["pub_decade"]  = ((df["pub_year"] // 10) * 10).astype("Int64")

    # Clean subject display column: first matching scope subject per paper
    df["subject_clean"] = df["Subject"].apply(_primary_subject)

    # Drop impossible (negative) time-to-retraction rows
    before = len(df)
    df = df[~(df["time_to_retraction_days"] < 0)].copy()
    dropped = before - len(df)

    print(f"  Dropped {dropped} rows with negative time-to-retraction")
    print(f"  Clean records: {len(df):,}  |  Hindawi papers: {df['is_hindawi'].sum():,}")
    if df["time_to_retraction_years"].notna().any():
        print(f"  Median time-to-retraction: {df['time_to_retraction_years'].median():.2f} years")
    return df


def _primary_subject(subject_str) -> str | None:
    """Return the display name of the first in-scope subject tag for a paper."""
    if pd.isna(subject_str):
        return None
    tags = [t.strip() for t in str(subject_str).split(";")]
    for scope_tag, display in SUBJECT_DISPLAY_NAMES.items():
        if any(scope_tag.lower() in t.lower() for t in tags):
            return display
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_and_filter()
    if df is None:   # EXPLORE_SUBJECTS mode
        return
    df = clean_and_engineer(df)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Parquet preserves the ReasonCategories list column and dtypes.
    df.to_parquet(CLEAN_PATH, index=False)
    print(f"\n  Saved cleaned dataset -> {CLEAN_PATH.relative_to(CLEAN_PATH.parent.parent)}")


def load_clean() -> pd.DataFrame:
    """Load the cleaned parquet produced by main(). Used by later stages."""
    if not CLEAN_PATH.exists():
        sys.exit("clean_retractions.parquet not found. Run: python src/data_prep.py")
    return pd.read_parquet(CLEAN_PATH)


if __name__ == "__main__":
    main()
