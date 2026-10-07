"""
descriptive.py
==============
Descriptive analyses: temporal trends, retraction reasons, geographic
distribution (with SCImago normalisation), and journal/publisher/author
breakdowns.

    python src/descriptive.py

Depends on data/clean_retractions.parquet (produced by data_prep.py).
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import (
    PALETTE, SCIMAGO_PATH, MAJOR_NATION_MIN_DOCS,
)
from utils import set_style, savefig, save_table, make_iso_resolver, build_reasons_long
from data_prep import load_clean


# ─────────────────────────────────────────────────────────────────────────────
#  TEMPORAL TRENDS
# ─────────────────────────────────────────────────────────────────────────────

def temporal_trends(df: pd.DataFrame) -> None:
    print("\n== Temporal trends ==")
    set_style()

    annual = (
        df.groupby("retraction_year").size().reset_index(name="count")
        .query("1980 <= retraction_year <= 2026")
    )
    annual_nh = (
        df[~df["is_hindawi"]].groupby("retraction_year").size()
        .reset_index(name="count_excl_hindawi")
        .query("1980 <= retraction_year <= 2026")
    )
    annual = annual.merge(annual_nh, on="retraction_year", how="left")
    save_table(annual, "annual_retractions")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(annual["retraction_year"], annual["count"], alpha=0.12, color=PALETTE[0])
    ax.plot(annual["retraction_year"], annual["count"],
            color=PALETTE[0], lw=2.5, marker="o", ms=4, label="All retractions")
    ax.plot(annual["retraction_year"], annual["count_excl_hindawi"],
            color=PALETTE[3], lw=2, ls="--", marker="s", ms=3, label="Excluding Hindawi")
    ax.axvline(2023, color=PALETTE[4], lw=1.2, ls="--", alpha=0.8, label="Hindawi mass retraction")
    ax.set_xlabel("Year of retraction")
    ax.set_ylabel("Number of retractions")
    ax.set_title("Annual biomedical retractions (1980-2026)")
    ax.legend(fontsize=9)
    savefig("fig01_annual_retractions")

    # Time-to-retraction distribution + by decade
    ttr = df["time_to_retraction_years"].dropna()
    ttr = ttr[(ttr >= 0) & (ttr <= 30)]
    dec = (
        df.assign(decade=df["pub_decade"].astype(str) + "s")
        [["decade", "time_to_retraction_years"]].dropna()
        .query("0 <= time_to_retraction_years <= 30")
    )
    order = sorted(dec["decade"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(ttr, bins=45, color=PALETTE[1], edgecolor="white", lw=0.4, alpha=0.9)
    axes[0].axvline(ttr.median(), color=PALETTE[4], lw=2, ls="--", label=f"Median: {ttr.median():.1f} yr")
    axes[0].axvline(ttr.mean(),   color=PALETTE[0], lw=2, ls=":",  label=f"Mean: {ttr.mean():.1f} yr")
    axes[0].set_xlabel("Time to retraction (years)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Time-to-retraction distribution")
    axes[0].legend()
    sns.violinplot(data=dec, x="decade", y="time_to_retraction_years",
                   order=order, palette=PALETTE[:len(order)],
                   inner="quartile", linewidth=0.8, ax=axes[1])
    axes[1].set_xlabel("Publication decade")
    axes[1].set_ylabel("Time to retraction (years)")
    axes[1].set_title("Time to retraction by publication decade")
    savefig("fig02_time_to_retraction")


# ─────────────────────────────────────────────────────────────────────────────
#  REASONS
# ─────────────────────────────────────────────────────────────────────────────

def reasons_analysis(df: pd.DataFrame, reasons_long: pd.DataFrame) -> None:
    print("\n== Reasons analysis ==")
    set_style()

    freq = reasons_long["Category"].value_counts()
    order = freq.index.tolist()
    colours = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(order)}
    save_table(freq.reset_index().rename(columns={"index": "Category", "Category": "count"}),
               "reason_frequency")

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(order[::-1], freq.values[::-1],
                   color=[colours[c] for c in order[::-1]], alpha=0.88)
    ax.bar_label(bars, labels=[f"{v:,}" for v in freq.values[::-1]], padding=5, fontsize=9.5)
    ax.set_xlim(0, freq.max() * 1.18)
    ax.set_xlabel("Number of retractions")
    ax.set_title("Retraction reasons by frequency")
    savefig("fig03_reason_frequency")

    # Co-occurrence matrix
    cooc = pd.DataFrame(0, index=order, columns=order)
    for cats in df["ReasonCategories"]:
        for a in cats:
            for b in cats:
                if a in cooc.index and b in cooc.columns:
                    cooc.loc[a, b] += 1
    np.fill_diagonal(cooc.values, 0)
    mask = np.triu(np.ones_like(cooc, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cooc, mask=mask, cmap="YlOrBr", annot=True, fmt="d",
                linewidths=0.4, ax=ax, cbar_kws={"label": "Co-occurrences", "shrink": 0.55})
    ax.set_title("Reason co-occurrence matrix")
    plt.xticks(rotation=40, ha="right")
    savefig("fig04_reason_cooccurrence")

    # Trends over time for the top 6 reasons
    top6 = order[:6]
    trend = (
        reasons_long[reasons_long["Category"].isin(top6)]
        .groupby(["retraction_year", "Category"]).size().reset_index(name="count")
        .query("1995 <= retraction_year <= 2023")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cat in enumerate(top6):
        sub = trend[trend["Category"] == cat]
        ax.plot(sub["retraction_year"], sub["count"],
                label=textwrap.shorten(cat, 32), color=PALETTE[i], lw=2.2, marker="o", ms=3)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of retractions")
    ax.set_title("Top-6 reason categories over time")
    ax.legend(fontsize=8.5, ncol=2)
    savefig("fig05_reason_trends")


# ─────────────────────────────────────────────────────────────────────────────
#  GEOGRAPHIC
# ─────────────────────────────────────────────────────────────────────────────

def geographic_analysis(df: pd.DataFrame) -> pd.DataFrame | None:
    print("\n== Geographic analysis ==")
    try:
        to_iso3, to_name = make_iso_resolver()
    except ImportError as exc:
        print(f"  {exc}")
        return None
    set_style()

    geo = (
        df[["Record ID", "Country", "retraction_year", "ReasonCategories"]]
        .assign(Country=df["Country"].str.split(";")).explode("Country")
    )
    geo["Country"] = geo["Country"].str.strip()
    invalid = {"unknown", "not available", "na", "n/a", "none", ""}
    geo = geo[~geo["Country"].str.lower().isin(invalid) & geo["Country"].notna()].copy()
    geo["iso3"] = geo["Country"].apply(to_iso3)
    geo = geo[geo["iso3"].notna()].copy()

    counts = geo.groupby("iso3").size().reset_index(name="retraction_count")
    counts["country_name"] = counts["iso3"].apply(to_name)

    # Choropleth (sqrt scale so one huge country does not wash out the map)
    _choropleth(counts, to_name)

    # Raw count bar chart
    top_raw = counts.nlargest(20, "retraction_count")
    fig, ax = plt.subplots(figsize=(9, 8))
    bars = ax.barh(top_raw["country_name"][::-1].values,
                   top_raw["retraction_count"][::-1].values, color=PALETTE[0], alpha=0.85)
    ax.bar_label(bars, labels=[f"{v:,}" for v in top_raw["retraction_count"][::-1].values],
                 padding=4, fontsize=9)
    ax.set_xlim(0, top_raw["retraction_count"].max() * 1.18)
    ax.set_xlabel("Number of retractions (raw count)")
    ax.set_title("Top 20 countries by raw retraction count")
    savefig("fig06_top_countries_raw")

    # Normalised rate for major research nations
    counts = _normalised_rates(counts, to_iso3, to_name, geo)

    # Reason profile per top-10 country
    top10 = counts.nlargest(10, "retraction_count")["iso3"].tolist()
    profile = (
        geo[geo["iso3"].isin(top10)]
        .explode("ReasonCategories").rename(columns={"ReasonCategories": "Category"})
    )
    ct = profile.groupby(["iso3", "Category"]).size().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.index = ct_pct.index.map(to_name)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(ct_pct, cmap="YlOrRd", annot=True, fmt=".0f",
                linewidths=0.4, ax=ax, cbar_kws={"label": "% of country's retractions", "shrink": 0.55})
    ax.set_title("Retraction reason profile by country (top 10 by raw count)")
    plt.xticks(rotation=38, ha="right", fontsize=9)
    savefig("fig08_country_reason_heatmap")

    save_table(counts, "country_counts")
    return counts


def _choropleth(counts: pd.DataFrame, to_name) -> None:
    try:
        import plotly.express as px
    except ImportError:
        print("  plotly not installed; skipping choropleth.")
        return
    cc = counts.copy()
    cc["sqrt_count"] = np.sqrt(cc["retraction_count"])
    fig = px.choropleth(
        cc, locations="iso3", color="sqrt_count",
        color_continuous_scale="Blues", hover_name="country_name",
        custom_data=["retraction_count"],
        title="Global biomedical retractions (colour = sqrt count; hover = raw count)",
    )
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Retractions: %{customdata[0]:,}<extra></extra>"
    )
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor="#AAAAAA"),
        font_family="Arial", title_font_size=14, margin=dict(l=0, r=0, t=50, b=10),
    )
    from config import FIGURES_DIR
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(FIGURES_DIR / "fig07_choropleth.html")
    print("    -> figures/fig07_choropleth.html")


def _normalised_rates(counts, to_iso3, to_name, geo) -> pd.DataFrame:
    if not SCIMAGO_PATH.exists():
        print(f"  {SCIMAGO_PATH.name} not found; skipping normalised rates.")
        return counts
    try:
        sc = pd.read_csv(SCIMAGO_PATH)
        sc["iso3"] = sc["Country"].apply(to_iso3)
        sc = sc.dropna(subset=["iso3"])
        docs = sc.groupby("iso3")["Documents"].sum().reset_index()
        major = docs[docs["Documents"] >= MAJOR_NATION_MIN_DOCS]
        print(f"  Major research nations (>= {MAJOR_NATION_MIN_DOCS:,} docs): {len(major)}")

        merged = counts.merge(major, on="iso3", how="inner")
        merged["rate_per_10k"] = merged["retraction_count"] / merged["Documents"] * 10_000

        def ci(row, n=2000):
            p = row["retraction_count"] / row["Documents"]
            sim = np.random.binomial(int(row["Documents"]), p, n) / row["Documents"] * 10_000
            return np.percentile(sim, [2.5, 97.5])

        cis = merged.apply(ci, axis=1)
        merged["ci_low"]  = [c[0] for c in cis]
        merged["ci_high"] = [c[1] for c in cis]
        merged["country_name"] = merged["iso3"].apply(to_name)

        top = merged.nlargest(20, "rate_per_10k")
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.barh(top["country_name"][::-1].values, top["rate_per_10k"][::-1].values,
                color=PALETTE[1], alpha=0.85)
        ax.errorbar(
            top["rate_per_10k"][::-1].values, top["country_name"][::-1].values,
            xerr=[(top["rate_per_10k"] - top["ci_low"])[::-1].values,
                  (top["ci_high"] - top["rate_per_10k"])[::-1].values],
            fmt="none", color="#333", lw=1.2, capsize=3,
        )
        for i, (_, row) in enumerate(top[::-1].reset_index(drop=True).iterrows()):
            ax.text(top["rate_per_10k"][::-1].iloc[i] * 1.01, i,
                    f"n={int(row['retraction_count']):,}", va="center", fontsize=8.5, color="#444")
        ax.set_xlim(0, top["rate_per_10k"].max() * 1.35)
        ax.set_xlabel("Retractions per 10,000 publications (95% CI)")
        ax.set_title(f"Top 20 countries by normalised rate\n(>= {MAJOR_NATION_MIN_DOCS:,} publications)")
        savefig("fig07b_top_countries_normalised")
        return merged
    except Exception as exc:
        print(f"  SCImago merge failed ({exc}); using raw counts.")
        return counts


# ─────────────────────────────────────────────────────────────────────────────
#  JOURNALS / PUBLISHERS / AUTHORS
# ─────────────────────────────────────────────────────────────────────────────

def journals_authors(df: pd.DataFrame) -> None:
    print("\n== Journals, publishers & authors ==")
    set_style()

    top_j = df["Journal"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(9, 8))
    bars = ax.barh(top_j.index[::-1], top_j.values[::-1], color=PALETTE[2], alpha=0.85)
    ax.bar_label(bars, labels=[f"{v:,}" for v in top_j.values[::-1]], padding=4, fontsize=9)
    ax.set_xlim(0, top_j.max() * 1.18)
    ax.set_xlabel("Number of retractions")
    ax.set_title("Top 20 journals by retraction count")
    savefig("fig09_top_journals")

    top_p = df["Publisher"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top_p.index[::-1], top_p.values[::-1], color=PALETTE[3], alpha=0.85)
    ax.bar_label(bars, labels=[f"{v:,}" for v in top_p.values[::-1]], padding=4, fontsize=9)
    ax.set_xlim(0, top_p.max() * 1.18)
    ax.set_xlabel("Number of retractions")
    ax.set_title("Top 15 publishers by retraction count")
    savefig("fig10_top_publishers")

    # Serial retractors
    author_counts = df["Author"].dropna().str.split(";").explode().str.strip().value_counts()
    hist = author_counts.value_counts().sort_index().loc[lambda s: s.index <= 15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(hist.index, hist.values, color=PALETTE[0], alpha=0.85, edgecolor="white", lw=0.5)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Retractions per author")
    ax.set_ylabel("Number of authors (log scale)")
    ax.set_title("Distribution of per-author retraction counts")
    savefig("fig11_serial_retractors")

    # Retractions and TTR by author count
    ac = df[["author_count", "time_to_retraction_years"]].dropna()
    ac = ac[ac["author_count"] <= 20]
    bins   = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20]
    labels = ["1", "2", "3", "4", "5", "6-7", "8-10", "11-15", "16-20"]
    ac["group"] = pd.cut(ac["author_count"], bins=bins, labels=labels, right=True)
    stats = (
        ac.groupby("group", observed=True)
        .agg(count=("time_to_retraction_years", "size"),
             median_ttr=("time_to_retraction_years", "median"))
        .reset_index()
    )
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.bar(stats["group"].astype(str), stats["count"], color=PALETTE[0], alpha=0.72, label="Retractions")
    ax2.plot(stats["group"].astype(str), stats["median_ttr"],
             color=PALETTE[4], lw=2.5, marker="D", ms=6, label="Median TTR")
    ax1.set_xlabel("Number of authors")
    ax1.set_ylabel("Number of retractions", color=PALETTE[0])
    ax2.set_ylabel("Median time to retraction (years)", color=PALETTE[4])
    ax1.set_title("Retractions and time-to-retraction by author count")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    savefig("fig12_retractions_by_author_count")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_clean()
    reasons_long = build_reasons_long(df)
    temporal_trends(df)
    reasons_analysis(df, reasons_long)
    geographic_analysis(df)
    journals_authors(df)
    print("\n  Descriptive analyses complete.")


if __name__ == "__main__":
    main()
