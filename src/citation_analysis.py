"""
citation_analysis.py
====================
Citation-based analyses:
  - Citation count and citation velocity vs. time-to-retraction
  - High- vs low-citation time-to-retraction comparison
  - Sleeping-beauty detection (long time-to-retraction x high citations)
  - Post-retraction citation contamination

STAGE 1 FIX (contamination)
---------------------------
The previous contamination analysis produced meaningless ratios (e.g. 11,720x)
because papers with a near-zero pre-retraction window inflate the post/pre ratio,
and very recent retractions have an undersized post-window. This version:
  * requires a pre-retraction window >= CONTAMINATION_MIN_PRE_YEARS,
  * excludes retractions after CONTAMINATION_MAX_RETRACT_YEAR,
  * flags and exports the papers still cited faster after retraction.

    python src/citation_analysis.py

Depends on data/enriched_retractions.parquet.
"""

from __future__ import annotations

import textwrap
from urllib.parse import quote

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu

from config import (
    PALETTE, CONTACT_EMAIL, CONTAMINATION_MIN_PRE_YEARS,
    CONTAMINATION_MAX_RETRACT_YEAR, CONTAMINATION_CURRENT_YEAR,
)
from utils import set_style, savefig, save_table
from citation_enrichment import load_enriched

_HEADERS = {"User-Agent": f"retraction-analysis (mailto:{CONTACT_EMAIL})"}


# ─────────────────────────────────────────────────────────────────────────────
#  CITATION vs TTR
# ─────────────────────────────────────────────────────────────────────────────

def citation_vs_ttr(df: pd.DataFrame) -> None:
    print("\n== Citation count vs time-to-retraction ==")
    set_style()

    plot_df = (
        df[["citation_count", "time_to_retraction_years"]].dropna()
        .query("0 <= time_to_retraction_years <= 25")
    )
    if len(plot_df) < 30:
        print("  Too few citation-matched papers; skipping.")
        return
    q97 = plot_df["citation_count"].quantile(0.97)
    plot_df = plot_df[plot_df["citation_count"] <= q97]

    rho, p = spearmanr(plot_df["citation_count"], plot_df["time_to_retraction_years"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(plot_df["citation_count"], plot_df["time_to_retraction_years"],
               alpha=0.22, s=12, color=PALETTE[0], rasterized=True)
    _add_lowess(ax, plot_df["citation_count"], plot_df["time_to_retraction_years"])
    ax.set_xlabel("Citation count")
    ax.set_ylabel("Time to retraction (years)")
    ax.set_title(f"Citation count vs time to retraction\nSpearman rho={rho:.3f}, p={p:.2e}, n={len(plot_df):,}")
    savefig("fig19_citations_vs_ttr")

    # Citation velocity
    vel = df.copy()
    vel["years_since_pub"] = (pd.Timestamp("today") - vel["OriginalPaperDate"]).dt.days / 365.25
    vel = vel[vel["years_since_pub"] > 0.5]
    vel["citation_velocity"] = vel["citation_count"] / vel["years_since_pub"]
    vdf = (
        vel[["citation_velocity", "time_to_retraction_years"]].dropna()
        .query("0 <= time_to_retraction_years <= 25")
    )
    if len(vdf) >= 30:
        q97v = vdf["citation_velocity"].quantile(0.97)
        vdf = vdf[vdf["citation_velocity"] <= q97v]
        rv, pv = spearmanr(vdf["citation_velocity"], vdf["time_to_retraction_years"])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(vdf["citation_velocity"], vdf["time_to_retraction_years"],
                   alpha=0.22, s=12, color=PALETTE[1], rasterized=True)
        _add_lowess(ax, vdf["citation_velocity"], vdf["time_to_retraction_years"])
        ax.set_xlabel("Citation velocity (citations per year since publication)")
        ax.set_ylabel("Time to retraction (years)")
        ax.set_title(f"Citation velocity vs time to retraction\nSpearman rho={rv:.3f}, p={pv:.2e}")
        savefig("fig20_citation_velocity_vs_ttr")

    # High vs low citation TTR
    med = df["citation_count"].median()
    hi = df.loc[df["citation_count"] > med, "time_to_retraction_years"].dropna()
    lo = df.loc[df["citation_count"] <= med, "time_to_retraction_years"].dropna()
    hi = hi[(hi >= 0) & (hi <= 30)]
    lo = lo[(lo >= 0) & (lo <= 30)]
    if len(hi) > 10 and len(lo) > 10:
        U, p_mw = mannwhitneyu(hi, lo, alternative="two-sided")
        print(f"  High-citation median TTR: {hi.median():.2f} yr (n={len(hi):,})")
        print(f"  Low-citation median TTR:  {lo.median():.2f} yr (n={len(lo):,})")
        print(f"  Mann-Whitney p={p_mw:.3e}")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.violinplot([lo.values, hi.values], positions=[1, 2], showmedians=True, widths=0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"Low-citation\n(<={med:.0f}, n={len(lo):,})",
                            f"High-citation\n(>{med:.0f}, n={len(hi):,})"])
        ax.set_ylabel("Time to retraction (years)")
        ax.set_title(f"Time to retraction: high vs low citation\nMann-Whitney p={p_mw:.2e} | "
                     f"delta median = {hi.median()-lo.median():.2f} yr")
        savefig("fig21_high_vs_low_citation_ttr")


def _add_lowess(ax, x, y):
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm = lowess(y, x, frac=0.3)
        ax.plot(sm[:, 0], sm[:, 1], color=PALETTE[4], lw=2.5, label="LOWESS")
        ax.legend(fontsize=9)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  SLEEPING BEAUTIES
# ─────────────────────────────────────────────────────────────────────────────

def sleeping_beauties(df: pd.DataFrame) -> pd.DataFrame | None:
    print("\n== Sleeping beauty analysis ==")
    set_style()

    sb = df[[
        "Record ID", "Title", "citation_count", "time_to_retraction_years",
        "ReasonCategories", "Journal", "retraction_year",
    ]].dropna(subset=["citation_count", "time_to_retraction_years"]).copy()
    sb = sb.query("0 < time_to_retraction_years <= 30 and citation_count > 0")
    if len(sb) < 30:
        print("  Too few papers; skipping.")
        return None

    sb["ttr_norm"] = (sb["time_to_retraction_years"] - sb["time_to_retraction_years"].min()) / \
                     (sb["time_to_retraction_years"].max() - sb["time_to_retraction_years"].min())
    q95 = sb["citation_count"].quantile(0.95)
    sb["cit_norm"] = sb["citation_count"].clip(upper=q95) / q95
    sb["sb_score"] = (sb["ttr_norm"] * sb["cit_norm"]) ** 0.5

    top = sb.nlargest(20, "sb_score").copy()
    save_table(
        top[["Title", "Journal", "citation_count", "time_to_retraction_years", "sb_score"]],
        "sleeping_beauties_top20",
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(sb["time_to_retraction_years"], sb["citation_count"].clip(upper=q95),
                    c=sb["sb_score"], cmap="YlOrRd", alpha=0.45, s=18, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Sleeping-beauty score (high TTR x high citations)")
    for _, row in top.head(8).iterrows():
        ax.annotate(textwrap.shorten(str(row["Title"]), 30),
                    (row["time_to_retraction_years"], min(row["citation_count"], q95)),
                    fontsize=6.5, color="#333",
                    xytext=(row["time_to_retraction_years"] + 0.5, min(row["citation_count"] * 1.1, q95)),
                    arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.7))
    ax.set_xlabel("Time to retraction (years)")
    ax.set_ylabel(f"Citation count (capped at 95th pct = {q95:.0f})")
    ax.set_title("Sleeping beauties: papers cited for years before retraction")
    savefig("fig22_sleeping_beauty")
    return top


# ─────────────────────────────────────────────────────────────────────────────
#  POST-RETRACTION CONTAMINATION  (STAGE 1 FIX)
# ─────────────────────────────────────────────────────────────────────────────

def post_retraction_contamination(df: pd.DataFrame) -> None:
    print("\n== Post-retraction citation contamination ==")
    set_style()

    # Contamination depth by reason (static proxy)
    cit = df[["citation_count", "ReasonCategories", "time_to_retraction_years"]].dropna(
        subset=["citation_count"]
    ).copy()
    cit = cit.query("citation_count > 0")
    cit["primary_reason"] = cit["ReasonCategories"].apply(lambda x: x[0] if len(x) else "Other / unknown")
    top6 = cit["primary_reason"].value_counts().index[:6]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=cit[cit["primary_reason"].isin(top6)],
                x="primary_reason", y="citation_count", order=top6,
                palette=PALETTE[:6], linewidth=0.8, fliersize=2, ax=ax)
    ax.set_yscale("log")
    ax.set_xticklabels([textwrap.shorten(t.get_text(), 22) for t in ax.get_xticklabels()],
                       rotation=28, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Citation count at retraction (log scale)")
    ax.set_title("Citation contamination depth by retraction reason")
    savefig("fig23_contamination_by_reason")

    # Pre vs post citation rate, with Stage 1 guardrails
    try:
        import requests
    except ImportError:
        print("  requests not installed; skipping time-series contamination.")
        return

    eligible = df[
        df["OriginalPaperDOI"].notna()
        & df["retraction_year"].notna()
        & df["pub_year"].notna()
        & df["citation_count"].notna()
    ].copy()
    # Guardrail 1: pre-retraction window must be long enough.
    eligible["pre_window"] = eligible["retraction_year"].astype(float) - eligible["pub_year"].astype(float)
    eligible = eligible[eligible["pre_window"] >= CONTAMINATION_MIN_PRE_YEARS]
    # Guardrail 2: exclude very recent retractions (undersized post-window).
    eligible = eligible[eligible["retraction_year"] <= CONTAMINATION_MAX_RETRACT_YEAR]

    sample = eligible.nlargest(60, "citation_count")
    print(f"  Fetching yearly citation curves for {len(sample)} eligible papers ...")

    rows = []
    for _, row in sample.iterrows():
        doi = str(row["OriginalPaperDOI"])
        ret_year = int(row["retraction_year"])
        pub_year = int(row["pub_year"])
        try:
            resp = requests.get(
                f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='')}",
                params={"select": "counts_by_year"}, headers=_HEADERS, timeout=25,
            )
            if resp.status_code != 200:
                continue
            cby = resp.json().get("counts_by_year", [])
            if not cby:
                continue
            ydf = pd.DataFrame(cby)
            pre  = ydf[ydf["year"] < ret_year]["cited_by_count"].sum()
            post = ydf[ydf["year"] >= ret_year]["cited_by_count"].sum()
            n_pre  = max(1, ret_year - pub_year)
            n_post = max(1, CONTAMINATION_CURRENT_YEAR - ret_year)
            rows.append({
                "title": row["Title"], "journal": row["Journal"], "doi": doi,
                "primary_reason": row["ReasonCategories"][0] if len(row["ReasonCategories"]) else "Other",
                "pub_year": pub_year, "retraction_year": ret_year,
                "citation_count": row["citation_count"],
                "pre_rate": pre / n_pre, "post_rate": post / n_post,
            })
        except Exception:
            pass

    if not rows:
        print("  No time-series data retrieved.")
        return

    rate_df = pd.DataFrame(rows)
    rate_df["ratio"] = rate_df["post_rate"] / (rate_df["pre_rate"] + 0.01)
    above = rate_df[rate_df["post_rate"] > rate_df["pre_rate"]].sort_values("ratio", ascending=False)
    print(f"  Papers still cited faster after retraction: {len(above)} / {len(rate_df)}")
    save_table(rate_df.sort_values("ratio", ascending=False), "post_retraction_rates")
    save_table(above, "papers_cited_faster_after_retraction")

    max_val = rate_df[["pre_rate", "post_rate"]].max().max() * 1.05
    colours = ["#E76F51" if p > r else "#2A9D8F"
               for p, r in zip(rate_df["post_rate"], rate_df["pre_rate"])]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(rate_df["pre_rate"], rate_df["post_rate"], c=colours, alpha=0.8, s=55, zorder=3)
    ax.plot([0, max_val], [0, max_val], color="#888", lw=1.2, ls="--", label="Pre = post rate")
    for _, row in above.head(6).iterrows():
        ax.annotate(textwrap.shorten(str(row["title"]), 28),
                    (row["pre_rate"], row["post_rate"]), fontsize=6.5, color="#333",
                    xytext=(row["pre_rate"] + max_val * 0.02, row["post_rate"] + max_val * 0.01),
                    arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.8))
    above_patch = mpatches.Patch(color="#E76F51", label=f"Cited faster after ({len(above)})")
    below_patch = mpatches.Patch(color="#2A9D8F", label=f"Citation rate dropped ({len(rate_df)-len(above)})")
    ax.legend(handles=[above_patch, below_patch], fontsize=9, loc="upper left")
    ax.set_xlabel("Pre-retraction citation rate (citations/year)")
    ax.set_ylabel("Post-retraction citation rate (citations/year)")
    ax.set_title("Post- vs pre-retraction citation rate\n(eligible papers only: pre-window >= "
                 f"{CONTAMINATION_MIN_PRE_YEARS:.0f} yr, retracted <= {CONTAMINATION_MAX_RETRACT_YEAR})")
    savefig("fig24_post_retraction_contamination")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_enriched()
    if df["citation_count"].notna().sum() == 0:
        print("  No citation data available. Run citation_enrichment.py first.")
        return
    citation_vs_ttr(df)
    sleeping_beauties(df)
    post_retraction_contamination(df)
    print("\n  Citation analyses complete.")


if __name__ == "__main__":
    main()
