"""
subject_analysis.py
===================
Per-subject breakdown of the study fields (Neuroscience, Biostatistics/
Epidemiology, Radiology/Imaging, Nanotechnology): retraction counts, median
time-to-retraction, reason profiles, and annual trends. If a per-subject
publication denominator is supplied (see config.SUBJECT_DENOM_PATH), also
computes retractions per 10,000 publications so fields are compared fairly.

    python src/subject_analysis.py

Depends on data/clean_retractions.parquet.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PALETTE, SUBJECT_DENOM_PATH
from utils import set_style, savefig, save_table
from data_prep import load_clean


def per_subject(df: pd.DataFrame) -> None:
    print("\n== Per-subject analysis ==")
    set_style()

    d = df[df["subject_clean"].notna()].copy()
    subjects = d["subject_clean"].value_counts().index.tolist()
    total = len(df)

    # Summary table: count, % of total, median TTR
    rows = []
    for s in subjects:
        sub = d[d["subject_clean"] == s]
        rows.append({
            "subject": s,
            "count": sub["Record ID"].nunique(),
            "pct_total": sub["Record ID"].nunique() / total * 100,
            "median_ttr": sub["time_to_retraction_years"].dropna().median(),
        })
    summary = pd.DataFrame(rows).sort_values("count", ascending=False)
    save_table(summary, "subject_summary")
    print(summary.to_string(index=False))

    # Optional denominator merge -> rate per 10k
    if SUBJECT_DENOM_PATH.exists():
        denom = pd.read_csv(SUBJECT_DENOM_PATH)   # columns: subject, publications
        summary = summary.merge(denom, on="subject", how="left")
        summary["rate_per_10k"] = summary["count"] / summary["publications"] * 10_000
        save_table(summary, "subject_summary_normalised")

    # Overview: count + median TTR side by side
    plot = summary.sort_values("count")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(plot["subject"], plot["count"], color=PALETTE[0], alpha=0.85)
    axes[0].set_xlabel("Number of retractions")
    axes[0].set_title("Retraction count per subject")
    axes[1].barh(plot["subject"], plot["median_ttr"], color=PALETTE[2], alpha=0.85)
    axes[1].set_xlabel("Median time to retraction (years)")
    axes[1].set_title("Median time to retraction per subject")
    savefig("fig25_subject_overview")

    # Reason profile per subject (% within subject)
    sr = (
        d.explode("ReasonCategories").rename(columns={"ReasonCategories": "Category"})
    )
    ct = sr.groupby(["subject_clean", "Category"]).size().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(15, max(4, len(ct_pct) * 0.8)))
    sns.heatmap(ct_pct, cmap="Blues", annot=True, fmt=".0f", linewidths=0.4, ax=ax,
                cbar_kws={"label": "% of subject's retractions", "shrink": 0.5})
    ax.set_title("Retraction reason profile by subject area")
    plt.xticks(rotation=38, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    savefig("fig26_subject_reason_heatmap")

    # Annual trends per subject
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, s in enumerate(subjects):
        yr = (
            d[d["subject_clean"] == s].groupby("retraction_year").size()
            .reset_index(name="count").query("1995 <= retraction_year <= 2023")
        )
        ax.plot(yr["retraction_year"], yr["count"], label=s,
                color=PALETTE[i], lw=2.2, marker="o", ms=3)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of retractions")
    ax.set_title("Annual retractions by subject area")
    ax.legend(fontsize=9)
    savefig("fig27_subject_annual_trends")


def main() -> None:
    df = load_clean()
    per_subject(df)
    print("\n  Per-subject analysis complete.")


if __name__ == "__main__":
    main()
