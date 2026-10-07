"""
statistics_models.py
====================
Inferential statistics and survival modelling.

STAGE 4 UPGRADES
----------------
1. Pairwise log-rank across ALL reason categories with Bonferroni correction
   (replaces a single arbitrary two-group comparison), rendered as a
   significance heatmap.
2. Cox proportional-hazards regression for time-to-retraction, giving
   interpretable hazard ratios for reason category and citation count
   (replaces the crash-prone OLS coefficient extraction).
3. A clean OLS variance-decomposition that reports how much time-to-retraction
   variance is explained by reason vs. country, with robust standard errors
   and a guarded coefficient table (the previous KeyError is fixed).
4. Bootstrap confidence interval on the median time-to-retraction.

    python src/statistics_models.py

Depends on data/enriched_retractions.parquet (or clean data if unavailable).
"""

from __future__ import annotations

import textwrap
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal

from config import (
    PALETTE, TTR_MIN_YEARS, TTR_MAX_YEARS, MIN_PAPERS_PER_COUNTRY,
    BOOTSTRAP_N, RANDOM_SEED,
)
from utils import set_style, savefig, save_table, make_iso_resolver, build_reasons_long
from citation_enrichment import load_enriched


# ─────────────────────────────────────────────────────────────────────────────
#  KAPLAN-MEIER + PAIRWISE LOG-RANK  (Stage 4.1)
# ─────────────────────────────────────────────────────────────────────────────

def survival_by_reason(df: pd.DataFrame, reasons_long: pd.DataFrame) -> None:
    print("\n== Survival analysis by reason ==")
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        print("  lifelines not installed; skipping survival analysis.")
        return
    set_style()

    top = reasons_long["Category"].value_counts().index[:6].tolist()
    km_df = (
        reasons_long[reasons_long["Category"].isin(top)]
        [["Record ID", "Category", "time_to_retraction_years"]]
        .drop_duplicates("Record ID")
        .dropna(subset=["time_to_retraction_years"])
        .query(f"{TTR_MIN_YEARS} <= time_to_retraction_years <= {TTR_MAX_YEARS}")
    )

    # KM curves
    fig, ax = plt.subplots(figsize=(9, 6))
    kmf = KaplanMeierFitter()
    for i, cat in enumerate(top):
        sub = km_df[km_df["Category"] == cat]["time_to_retraction_years"]
        if len(sub) < 10:
            continue
        kmf.fit(sub, label=textwrap.shorten(cat, 32))
        kmf.plot_survival_function(ax=ax, ci_show=True, color=PALETTE[i], lw=2.2)
    ax.set_xlabel("Time to retraction (years)")
    ax.set_ylabel("Proportion not yet retracted")
    ax.set_title("Kaplan-Meier curves by reason category")
    ax.legend(fontsize=9)
    savefig("fig13_km_survival")

    # Pairwise log-rank with Bonferroni correction
    cats = [c for c in top if (km_df["Category"] == c).sum() >= 10]
    pmat = pd.DataFrame(np.nan, index=cats, columns=cats)
    n_tests = len(list(combinations(cats, 2)))
    for a, b in combinations(cats, 2):
        ta = km_df[km_df["Category"] == a]["time_to_retraction_years"]
        tb = km_df[km_df["Category"] == b]["time_to_retraction_years"]
        p = logrank_test(ta, tb).p_value
        p_adj = min(1.0, p * n_tests)   # Bonferroni
        pmat.loc[a, b] = p_adj
        pmat.loc[b, a] = p_adj
    np.fill_diagonal(pmat.values, 1.0)
    save_table(pmat.reset_index(), "pairwise_logrank_bonferroni", index=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    labels = pmat.copy()
    annot = labels.applymap(lambda v: "n.s." if v >= 0.05 else (f"{v:.1e}" if v < 1e-3 else f"{v:.3f}"))
    sns.heatmap(-np.log10(pmat.clip(lower=1e-300)), annot=annot, fmt="",
                cmap="RdYlGn", linewidths=0.5, ax=ax,
                cbar_kws={"label": "-log10(Bonferroni p)", "shrink": 0.6},
                xticklabels=[textwrap.shorten(c, 18) for c in cats],
                yticklabels=[textwrap.shorten(c, 18) for c in cats])
    ax.set_title("Pairwise log-rank tests (Bonferroni-corrected)\ngreen = significant difference in TTR")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    savefig("fig14_pairwise_logrank")


# ─────────────────────────────────────────────────────────────────────────────
#  COX PROPORTIONAL HAZARDS  (Stage 4.2)
# ─────────────────────────────────────────────────────────────────────────────

def cox_model(df: pd.DataFrame) -> None:
    print("\n== Cox proportional-hazards model ==")
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        print("  lifelines not installed; skipping Cox model.")
        return
    set_style()

    cox_df = df[[
        "time_to_retraction_years", "ReasonCategories", "citation_count",
        "author_count", "retraction_year",
    ]].copy()
    cox_df = cox_df.dropna(subset=["time_to_retraction_years"])
    cox_df = cox_df.query(f"{TTR_MIN_YEARS} < time_to_retraction_years <= {TTR_MAX_YEARS}")

    # Primary reason = first category; one-hot encode the top reasons.
    cox_df["primary_reason"] = cox_df["ReasonCategories"].apply(lambda x: x[0] if len(x) else "Other / unknown")
    top_reasons = cox_df["primary_reason"].value_counts().index[:6].tolist()
    cox_df = cox_df[cox_df["primary_reason"].isin(top_reasons)]

    # Model covariates: log citation count, author count, year, reason dummies.
    # Guard against columns that are entirely missing (median() would be NaN).
    cit_median = cox_df["citation_count"].median()
    if pd.isna(cit_median):
        print("  No citation data available; fitting Cox without citation covariate.")
        cox_df["log_citation"] = 0.0
    else:
        cox_df["log_citation"] = np.log1p(cox_df["citation_count"].fillna(cit_median))

    auth_median = cox_df["author_count"].median()
    cox_df["author_count"] = cox_df["author_count"].fillna(auth_median if not pd.isna(auth_median) else 1)
    cox_df["year_c"] = cox_df["retraction_year"].astype(float) - cox_df["retraction_year"].astype(float).mean()

    design = pd.get_dummies(cox_df[["primary_reason"]], drop_first=True)
    model_df = pd.concat([
        cox_df[["time_to_retraction_years", "log_citation", "author_count", "year_c"]].reset_index(drop=True),
        design.reset_index(drop=True).astype(float),
    ], axis=1)
    # lifelines needs an event indicator; every paper here is retracted -> event=1.
    model_df["event"] = 1
    # Drop any constant (zero-variance) covariate columns to aid convergence.
    covariate_cols = [c for c in model_df.columns if c not in ("time_to_retraction_years", "event")]
    constant = [c for c in covariate_cols if model_df[c].nunique() <= 1]
    if constant:
        model_df = model_df.drop(columns=constant)
    model_df = model_df.dropna()
    if len(model_df) < 30 or model_df.shape[1] <= 2:
        print("  Not enough usable data for Cox model; skipping.")
        return

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(model_df, duration_col="time_to_retraction_years", event_col="event")
    except Exception as exc:
        print(f"  Cox model failed to converge: {exc}")
        return

    summary = cph.summary[["coef", "exp(coef)", "coef lower 95%", "coef upper 95%", "p"]].copy()
    summary = summary.rename(columns={"exp(coef)": "hazard_ratio"})
    save_table(summary.reset_index(), "cox_summary", index=False)
    print(f"  Cox model concordance: {cph.concordance_index_:.3f}")

    # Forest plot of hazard ratios
    hr = summary.copy()
    hr["hr_low"]  = np.exp(hr["coef lower 95%"])
    hr["hr_high"] = np.exp(hr["coef upper 95%"])
    hr = hr.sort_values("hazard_ratio")
    nice = {
        "log_citation": "Log citation count",
        "author_count": "Author count",
        "year_c": "Retraction year (centred)",
    }
    hr.index = [nice.get(i, i.replace("primary_reason_", "")) for i in hr.index]

    fig, ax = plt.subplots(figsize=(9, max(5, len(hr) * 0.5)))
    colours = [PALETTE[1] if p < 0.05 else PALETTE[7] for p in hr["p"]]
    ax.errorbar(hr["hazard_ratio"], range(len(hr)),
                xerr=[hr["hazard_ratio"] - hr["hr_low"], hr["hr_high"] - hr["hazard_ratio"]],
                fmt="o", color="#333", ecolor="#999", capsize=3, ms=0, zorder=1)
    ax.scatter(hr["hazard_ratio"], range(len(hr)), c=colours, s=70, zorder=2)
    ax.axvline(1.0, color="#555", lw=1.2, ls="--")
    ax.set_yticks(range(len(hr)))
    ax.set_yticklabels(hr.index)
    ax.set_xlabel("Hazard ratio (>1 = retracted faster)")
    ax.set_title(f"Cox proportional-hazards model of time to retraction\n(concordance = {cph.concordance_index_:.3f})")
    sig = mpatches.Patch(color=PALETTE[1], label="p < 0.05")
    ns  = mpatches.Patch(color=PALETTE[7], label="p >= 0.05")
    ax.legend(handles=[sig, ns], fontsize=9)
    savefig("fig15_cox_hazard_ratios")


# ─────────────────────────────────────────────────────────────────────────────
#  VARIANCE DECOMPOSITION: REASON vs COUNTRY  (Stage 4.3, OLS crash fixed)
# ─────────────────────────────────────────────────────────────────────────────

def variance_decomposition(df: pd.DataFrame) -> None:
    print("\n== Variance decomposition: reason vs country ==")
    try:
        import statsmodels.formula.api as smf
        to_iso3, to_name = make_iso_resolver()
    except ImportError as exc:
        print(f"  {exc}; skipping.")
        return
    set_style()

    d = df[["Country", "time_to_retraction_years", "ReasonCategories", "retraction_year"]].copy()
    d = d.dropna(subset=["Country", "time_to_retraction_years"])
    d = d.query(f"{TTR_MIN_YEARS} <= time_to_retraction_years <= {TTR_MAX_YEARS}")
    d["primary_country"] = d["Country"].str.split(";").apply(lambda x: x[0].strip() if x else None)
    d["iso3"] = d["primary_country"].apply(to_iso3)
    d = d.dropna(subset=["iso3"])
    d["primary_reason"] = d["ReasonCategories"].apply(lambda x: x[0] if len(x) else "Other / unknown")
    d["country_name"]   = d["iso3"].apply(to_name)

    counts = d["iso3"].value_counts()
    d = d[d["iso3"].isin(counts[counts >= MIN_PAPERS_PER_COUNTRY].index)].copy()
    print(f"  Countries with >= {MIN_PAPERS_PER_COUNTRY} papers: {d['iso3'].nunique()}")

    if d["iso3"].nunique() < 2:
        print("  Fewer than two countries meet the threshold; skipping this analysis.")
        return

    # Kruskal-Wallis across countries
    groups = [g["time_to_retraction_years"].values for _, g in d.groupby("iso3")]
    H, p_kw = kruskal(*groups)
    print(f"  Kruskal-Wallis across countries: H={H:.1f}, p={p_kw:.3e}")

    d["log_ttr"]      = np.log1p(d["time_to_retraction_years"])
    d["reason_code"]  = d["primary_reason"].astype("category").cat.codes
    d["year_c"]       = d["retraction_year"].astype(float) - d["retraction_year"].astype(float).mean()

    # Two nested OLS models; compare R^2 to attribute variance.
    m_reason  = smf.ols("log_ttr ~ C(primary_reason) + year_c", data=d).fit()
    m_full    = smf.ols("log_ttr ~ C(primary_reason) + C(country_name) + year_c", data=d).fit(cov_type="HC3")

    r2_reason = m_reason.rsquared
    r2_full   = m_full.rsquared
    delta     = r2_full - r2_reason
    print(f"  R2 reason + year            : {r2_reason:.3f}")
    print(f"  R2 reason + country + year  : {r2_full:.3f}")
    print(f"  Delta R2 attributable to country: {delta:.3f}")

    save_table(
        pd.DataFrame({
            "model": ["reason+year", "reason+country+year"],
            "r_squared": [r2_reason, r2_full],
            "kruskal_H": [H, H], "kruskal_p": [p_kw, p_kw],
            "delta_r2_country": [np.nan, delta],
        }),
        "variance_decomposition",
    )

    # Guarded country-coefficient extraction (this is what used to KeyError).
    params = m_full.params
    conf   = m_full.conf_int()
    pvals  = m_full.pvalues
    country_terms = [t for t in params.index if t.startswith("C(country_name)")]
    if country_terms:
        coef_df = pd.DataFrame({
            "term": [t.split("T.")[-1].rstrip("]") for t in country_terms],
            "coef": [params[t] for t in country_terms],
            "ci_low": [conf.loc[t, 0] for t in country_terms],
            "ci_high": [conf.loc[t, 1] for t in country_terms],
            "p": [pvals[t] for t in country_terms],
        }).sort_values("coef")
        coef_df["sig"] = coef_df["p"] < 0.05
        save_table(coef_df, "country_ols_coefficients")

        fig, ax = plt.subplots(figsize=(9, max(5, len(coef_df) * 0.4)))
        colours = [PALETTE[1] if s else PALETTE[7] for s in coef_df["sig"]]
        ax.errorbar(coef_df["coef"], range(len(coef_df)),
                    xerr=[coef_df["coef"] - coef_df["ci_low"], coef_df["ci_high"] - coef_df["coef"]],
                    fmt="o", color="#333", ecolor="#999", capsize=3, ms=0, zorder=1)
        ax.scatter(coef_df["coef"], range(len(coef_df)), c=colours, s=60, zorder=2)
        ax.axvline(0, color="#555", lw=1.2, ls="--")
        ax.set_yticks(range(len(coef_df)))
        ax.set_yticklabels(coef_df["term"])
        ax.set_xlabel("OLS coefficient on log(time-to-retraction)")
        ax.set_title(f"Country effect on time to retraction\n"
                     f"controlling for reason and year (delta R2 = {delta:.3f})")
        sig = mpatches.Patch(color=PALETTE[1], label="p < 0.05")
        ns  = mpatches.Patch(color=PALETTE[7], label="p >= 0.05")
        ax.legend(handles=[sig, ns], fontsize=9)
        savefig("fig16_country_coefficients")

    # Country x reason median TTR heatmap (shows reason dominates)
    top_c = d["iso3"].value_counts().index[:6].tolist()
    top_r = d["primary_reason"].value_counts().index[:6].tolist()
    strat = (
        d[d["iso3"].isin(top_c) & d["primary_reason"].isin(top_r)]
        .groupby(["country_name", "primary_reason"])["time_to_retraction_years"]
        .median().unstack()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(strat, cmap="RdYlGn_r", annot=True, fmt=".1f", linewidths=0.4, ax=ax,
                cbar_kws={"label": "Median TTR (years)", "shrink": 0.5})
    ax.set_title("Median time to retraction by country x reason\n(reason, not country, drives most variation)")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    savefig("fig17_country_reason_ttr")


# ─────────────────────────────────────────────────────────────────────────────
#  BOOTSTRAP + CHI-SQUARE  (Stage 4.4)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_and_chisquare(df: pd.DataFrame, reasons_long: pd.DataFrame) -> None:
    print("\n== Bootstrap CI + chi-square ==")
    set_style()
    rng = np.random.default_rng(RANDOM_SEED)

    ttr = df["time_to_retraction_years"].dropna().values
    ttr = ttr[(ttr >= 0) & (ttr <= 30)]
    boot = [np.median(rng.choice(ttr, len(ttr), replace=True)) for _ in range(BOOTSTRAP_N)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  Median TTR: {np.median(ttr):.2f} yr (95% CI {lo:.2f}-{hi:.2f})")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(boot, bins=60, color=PALETTE[1], edgecolor="white", lw=0.3, alpha=0.9)
    ax.axvline(np.median(ttr), color=PALETTE[4], lw=2, label=f"Median: {np.median(ttr):.2f} yr")
    ax.axvspan(lo, hi, alpha=0.15, color=PALETTE[4], label=f"95% CI [{lo:.2f}, {hi:.2f}]")
    ax.set_xlabel("Bootstrap median TTR (years)")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap distribution of median time to retraction")
    ax.legend(fontsize=9)
    savefig("fig18_bootstrap_median_ttr")

    # Chi-square: reason category x paywalled
    pr = reasons_long.drop_duplicates("Record ID").dropna(subset=["paywalled"])
    if pr["paywalled"].nunique() == 2:
        ct = pd.crosstab(pr["paywalled"], pr["Category"])
        chi2, p, dof, _ = chi2_contingency(ct)
        print(f"  Chi-square (reason x paywalled): chi2={chi2:.1f}, dof={dof}, p={p:.3e}")
        save_table(
            pd.DataFrame({"statistic": ["chi2", "dof", "p_value"], "value": [chi2, dof, p]}),
            "chisquare_reason_paywalled",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_enriched()
    reasons_long = build_reasons_long(df)
    survival_by_reason(df, reasons_long)
    cox_model(df)
    variance_decomposition(df)
    bootstrap_and_chisquare(df, reasons_long)
    print("\n  Statistical modelling complete.")


if __name__ == "__main__":
    main()
