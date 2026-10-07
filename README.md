# Retraction Patterns in Quantitative and Imaging-Focused Biomedical Fields

A reproducible analysis pipeline characterising retracted publications across four
quantitative, data- and image-intensive biomedical research areas — **Neuroscience,
Biostatistics/Epidemiology, Radiology/Imaging, and Nanotechnology** — using the
[Retraction Watch Database](http://retractiondatabase.org/).

The pipeline quantifies temporal trends, retraction reasons, geographic distribution,
journal and author patterns, time-to-retraction, and the relationship between a paper's
citation profile and how quickly it is retracted. It further examines "sleeping beauty"
retractions and post-retraction citation contamination.

---

## Overview

Retraction is the scientific community's mechanism for removing invalid work from the
record, yet retraction is often slow and incomplete. This project asks:

1. **What drives retraction in quantitative and imaging-heavy biomedical fields, and
   how do the fields differ from one another?**
2. **What predicts how long a paper survives before it is retracted?** In particular,
   does a paper's citation profile — its prominence in the literature — relate to the
   speed of its retraction?
3. **How much influence does retracted work retain after it is retracted?**

The analysis is organised as a set of independent, cached stages so that any part can be
re-run in isolation and the whole pipeline is reproducible from a single command.

---

## Data

The pipeline expects the following files in `data/` (none are redistributed here; see
the licensing note below):

| File | Source | Purpose |
|---|---|---|
| `retraction_watch.csv` | [Retraction Watch Database](http://retractiondatabase.org/) (CC BY via Crossref) | Primary dataset of retractions |
| `scimagojr-country.csv` *(optional)* | [SCImago Journal & Country Rank](https://www.scimagojr.com/countryrank.php) | Country publication counts, used to normalise retraction rates |
| `scimago_subject_totals.csv` *(optional)* | SCImago, by subject area | Per-field publication counts, used to normalise per-subject rates |

`scimago_subject_totals.csv`, if supplied, should have two columns: `subject`
(matching the display names in `config.py`) and `publications` (integer).

The study scope, reason-category mapping, and all analysis parameters are defined in
[`src/config.py`](src/config.py).

Citation counts are retrieved at run time from two public APIs:
[NIH iCite](https://icite.od.nih.gov/) (via PubMed ID) and
[OpenAlex](https://openalex.org/) (via DOI, then title). Results are cached to
`data/citation_cache.parquet` so the APIs are queried only once.

---

## Methods

**Scope and cleaning.** Records are restricted to `RetractionNature == "Retraction"`
and to the four subject areas above. Dates are parsed to compute time-to-retraction;
missing DOIs and PubMed IDs are normalised; author counts and a Hindawi-publisher flag
are derived. The ~90 raw Retraction Watch reason codes are collapsed into 13
interpretable categories (mapping in `config.py`).

**Descriptive analysis.** Annual retraction trends (with a Hindawi-excluded sensitivity
line), time-to-retraction distributions by decade, reason frequencies and co-occurrence,
a country choropleth (square-root colour scale to prevent a single high-volume country
from dominating), publication-normalised country rates restricted to major research
nations, and journal/publisher/author breakdowns.

**Citation enrichment.** Citation counts and NIH Relative Citation Ratios are matched
from iCite by PubMed ID, with a batched OpenAlex DOI lookup and a title-search fallback
to cover non-PubMed-indexed journals.

**Statistical modelling.**
- Kaplan–Meier time-to-retraction curves by reason, with **all-pairs log-rank tests**
  under Bonferroni correction.
- A **Cox proportional-hazards model** giving hazard ratios for reason category,
  citation count, author count, and year.
- A **variance decomposition** (nested OLS with HC3 robust errors) quantifying how much
  of the variation in time-to-retraction is attributable to reason versus country.
- Bootstrap confidence intervals and a reason-by-paywall chi-square test.

**Citation dynamics.** Citation count and citation velocity versus time-to-retraction;
a high- versus low-citation time-to-retraction comparison; sleeping-beauty scoring; and a
post-retraction contamination analysis comparing pre- and post-retraction citation rates
(restricted to papers with a sufficient pre-retraction window and excluding very recent
retractions, so the ratio is meaningful).

**Networks.** A keyword co-occurrence network from titles and a co-author network of
frequent retractors, both with Louvain community detection.

---

## Repository structure

```
.
├── run_all.sh                 # Runs the full pipeline end to end
├── Makefile                   # Convenience targets for each stage
├── requirements.txt
├── config.yaml                # Human-readable copy of key settings
├── docs/
│   └── index.html             # Interactive browser dashboard (GitHub Pages)
├── src/
│   ├── config.py              # All paths, scope, reason mapping, parameters
│   ├── utils.py               # Plotting style, figure/table saving, ISO helpers
│   ├── data_prep.py           # Stage 1: load, filter, clean, feature-engineer
│   ├── descriptive.py         # Stage 2: temporal, reasons, geographic, journals
│   ├── subject_analysis.py    # Stage 3: per-subject breakdowns
│   ├── citation_enrichment.py # Stage 4: iCite + OpenAlex citation matching
│   ├── citation_analysis.py   # Stage 5: citation vs TTR, sleeping beauties, contamination
│   ├── statistics_models.py   # Stage 6: survival, Cox, variance decomposition
│   └── network_analysis.py    # Stage 7: keyword and co-author networks
├── data/                      # Input CSVs + cached parquets (not committed)
├── figures/                   # Generated figures (PDF + PNG, and HTML for interactive)
└── tables/                    # Generated result tables (CSV)
```

Stages communicate through cached parquet files in `data/`, so each script can be run on
its own once `data_prep.py` has produced `clean_retractions.parquet`.

---

## Interactive dashboard (no install required)

A browser dashboard presents the main results directly, with no setup for the reader.
It opens showing the precomputed figures — headline counts, retractions per year,
time-to-retraction, reason categories, leading countries and publishers, retractions by
author count, time-to-retraction by country, and a per-field comparison — and a subject
dropdown switches between the four fields, the combined biomedical set, and all papers.

The dashboard reads a small precomputed `docs/summary.json` rather than the raw database,
so the page loads instantly and no large data file is redistributed. The summary is
produced from the Retraction Watch CSV by `build_summary.py`:

```bash
python build_summary.py path/to/retraction_watch.csv docs/summary.json
```

Regenerate `summary.json` whenever you refresh the database, then commit it alongside
`docs/index.html`.

**To view locally**, open `docs/index.html` in a browser (it needs `summary.json` in the
same folder).

**To publish on GitHub Pages:**

1. Push this repository to GitHub.
2. Go to **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to *Deploy from a branch*, choose your
   default branch, and set the folder to **`/docs`**.
4. Save. After a minute the dashboard is live at
   `https://<your-username>.github.io/<repo-name>/`, and anyone who opens the link sees
   the results immediately.

---

## How to run the full pipeline

```bash
# 1. Clone
git clone https://github.com/<your-username>/retraction-analysis.git
cd retraction-analysis

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the data
#    Place retraction_watch.csv (and optionally the SCImago CSVs) in data/

# 5. Set your contact email for the OpenAlex/NCBI polite pools
#    Edit CONTACT_EMAIL in src/config.py

# 6. Run the full pipeline
bash run_all.sh
```

Outputs are written to `figures/` and `tables/`.

**Running individual stages** (after `data_prep.py` has run once):

```bash
python src/descriptive.py
python src/statistics_models.py
make stats            # equivalent, via the Makefile
```

**Useful flags:**

```bash
bash run_all.sh --no-citations   # skip the API-dependent citation stages
bash run_all.sh --no-network     # skip the slower network stage
make clean                       # remove generated figures, tables, and caches
```

**Finding your subject tags.** To see the full list of Retraction Watch subject tags and
choose your own scope, set `EXPLORE_SUBJECTS = True` in `config.py` and run
`python src/data_prep.py`; it prints every tag and exits. Paste the ones you want into
`BIOMEDICAL_SUBJECTS`, set the flag back to `False`, and re-run.

---

## Reproducibility

- A global random seed (`config.RANDOM_SEED`) is set for all stochastic steps
  (bootstrap, synthetic sampling, network layouts).
- Citation API results are cached so repeated runs are deterministic and do not re-query
  the services.
- The Retraction Watch Database is updated continuously; record the download date when
  reporting results, since counts will change over time.

---

## Limitations

- **Coverage bias.** Retraction Watch skews toward English-language, PubMed/Web of
  Science–indexed literature, so rates for non-anglophone research systems are likely
  underestimated.
- **Detection versus occurrence.** The data reflect retractions that were *detected and
  acted upon*, not the true prevalence of flawed work.
- **The 2023 Hindawi event.** A single publisher's mass retraction in 2023 strongly
  influences several distributions; a Hindawi-excluded sensitivity line is reported
  alongside the main temporal trend.
- **Citation matching is incomplete and non-random**, tending to favour more prominent
  journals; citation-based results should be read with that in mind.
- **Reason categorisation** collapses a fine-grained controlled vocabulary into 13
  labels, and per-paper analyses use the first listed reason as the primary reason.

---

## License and attribution

The code in this repository is released under the MIT License. The Retraction Watch
Database is made available by The Center for Scientific Integrity through Crossref under a
CC BY license; please cite it accordingly and consult its terms before redistributing any
derived data. OpenAlex and NIH iCite data are used under their respective open terms.

---

## References

- The Center for Scientific Integrity. *The Retraction Watch Database.* Retraction Watch.
  http://retractiondatabase.org/
- Priem J, Piwowar H, Orr R. *OpenAlex: A fully-open index of scholarly works, authors,
  venues, institutions, and concepts.* 2022. https://openalex.org/
- Hutchins BI, et al. *Relative Citation Ratio (RCR): A new metric that uses citation
  rates to measure influence at the article level.* PLoS Biology, 2016.
- SCImago. *SJR — SCImago Journal & Country Rank.* https://www.scimagojr.com/
