"""
utils.py
========
Shared helpers used across the pipeline: plotting style, figure saving,
country-name-to-ISO3 resolution, and the long-form reason table builder.

Importing this module also applies the global matplotlib style and seeds
NumPy so that every stage is reproducible.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    PALETTE, FIG_DPI, FIGURES_DIR, TABLES_DIR, RANDOM_SEED,
)

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def set_style() -> None:
    """Apply the publication figure style globally."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": FIG_DPI,
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F8F6",
        "axes.edgecolor": "#BBBBBB",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "#E2E2E2",
        "grid.linewidth": 0.5,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.frameon": True,
    })


def savefig(name: str) -> None:
    """Save the current figure as both PDF (vector) and PNG (raster)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(FIGURES_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close("all")
    print(f"    -> figures/{name}.pdf / .png")


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    """Save a results table as CSV."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"{name}.csv", index=index)
    print(f"    -> tables/{name}.csv")


# ─────────────────────────────────────────────────────────────────────────────
#  COUNTRY / ISO3 RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

_ALIASES = {
    "United States": "USA", "USA": "USA", "U.S.A.": "USA",
    "UK": "GBR", "United Kingdom": "GBR",
    "China": "CHN", "People's Republic of China": "CHN",
    "Iran": "IRN", "South Korea": "KOR", "Republic of Korea": "KOR",
    "Taiwan": "TWN", "Russia": "RUS", "Czech Republic": "CZE",
    "Venezuela": "VEN", "Bolivia": "BOL",
}


def make_iso_resolver():
    """
    Return (to_iso3, to_name) functions that convert free-text country names
    to ISO-3166 alpha-3 codes and back, using pycountry with a fuzzy fallback.
    """
    try:
        import pycountry
        from thefuzz import process as fuzz_process
    except ImportError as exc:
        raise ImportError(
            "pycountry and thefuzz are required: pip install pycountry thefuzz"
        ) from exc

    name_to_iso3 = {c.name: c.alpha_3 for c in pycountry.countries}
    name_to_iso3.update({c.alpha_2: c.alpha_3 for c in pycountry.countries})
    name_to_iso3.update({c.alpha_3: c.alpha_3 for c in pycountry.countries})

    cache: dict[str, str | None] = {}

    def to_iso3(name):
        if pd.isna(name):
            return None
        name = str(name).strip()
        if name in cache:
            return cache[name]
        if name in _ALIASES:
            cache[name] = _ALIASES[name]
            return cache[name]
        try:
            result = pycountry.countries.lookup(name).alpha_3
        except LookupError:
            match, score = fuzz_process.extractOne(name, list(name_to_iso3.keys()))
            result = name_to_iso3[match] if score >= 90 else None
        cache[name] = result
        return result

    @lru_cache(maxsize=None)
    def to_name(code):
        try:
            return pycountry.countries.get(alpha_3=code).name
        except Exception:
            return code

    return to_iso3, to_name


# ─────────────────────────────────────────────────────────────────────────────
#  LONG-FORM REASON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_reasons_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the ReasonCategories list column so there is one row per
    (Record ID, reason category). All feature columns are carried along
    so downstream analyses (survival, chi-square) have what they need.
    """
    return (
        df[[
            "Record ID", "ReasonCategories", "retraction_year", "pub_year",
            "pub_decade", "paywalled", "time_to_retraction_years",
            "author_count", "is_hindawi",
        ]]
        .explode("ReasonCategories")
        .rename(columns={"ReasonCategories": "Category"})
    )
