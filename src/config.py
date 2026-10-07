"""
config.py
=========
Central configuration for the biomedical retraction analysis pipeline.

Every module imports its settings from here so the whole pipeline is driven by
one file. Paths are resolved relative to the repository root, so the code runs
identically regardless of the working directory it is launched from.
"""

from __future__ import annotations

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS  (resolved relative to the repo root, i.e. the parent of src/)
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR    = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
TABLES_DIR  = REPO_ROOT / "tables"

# Input files (place these in data/)
DATA_PATH       = DATA_DIR / "retraction_watch.csv"
SCIMAGO_PATH    = DATA_DIR / "scimagojr-country.csv"
# Optional: per-subject publication counts for denominators (see README)
SUBJECT_DENOM_PATH = DATA_DIR / "scimago_subject_totals.csv"

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS SCOPE
# ─────────────────────────────────────────────────────────────────────────────

# Set True once to print every unique Subject tag in the retraction-only subset,
# then paste the tags you want into BIOMEDICAL_SUBJECTS and set this back to False.
EXPLORE_SUBJECTS = False

# The subject areas that define the study scope. These are Retraction Watch
# subject tags (semicolon-separated in the raw data). Edit freely.
BIOMEDICAL_SUBJECTS = [
    "(BLS) Neuroscience",
    "(HSC) Biostatistics/Epidemiology",
    "(HSC) Radiology/Imaging",
    "(PHY) Nanotechnology",
]

# Human-readable labels for the subjects above (used in per-subject figures).
# Maps the raw tag (stripped) to a clean display name.
SUBJECT_DISPLAY_NAMES = {
    "(BLS) Neuroscience": "Neuroscience",
    "(HSC) Biostatistics/Epidemiology": "Biostatistics/Epidemiology",
    "(HSC) Radiology/Imaging": "Radiology/Imaging",
    "(PHY) Nanotechnology": "Nanotechnology",
}

# Only analyse rows whose RetractionNature is exactly this.
RETRACTION_NATURE = "Retraction"

# ─────────────────────────────────────────────────────────────────────────────
#  REASON CATEGORY MAPPING
#  Keys   = the 13 collapsed labels shown in every figure.
#  Values = the raw Retraction Watch reason strings that map to each label.
#  Any raw reason not listed here is mapped to "Other / unknown".
# ─────────────────────────────────────────────────────────────────────────────

REASON_CATEGORIES = {
    "Data and Results Issues": [
        "Unreliable Results", "Unreliable Results and/or Conclusions",
        "Concerns/Issues About Data", "Error in Data", "Original Data not Provided",
        "Unreliable Data", "Error in Results and/or Conclusions", "Error in Analyses",
        "Error in Methods", "Results Not Reproducible", "Concerns/Issues About Results",
        "Concerns/Issues about Results and/or Conclusions", "Concerns/Issues about Article",
        "Concerns/Issues about Methods", "Error in Text", "Error in Materials (General)",
        "Error in Materials", "Error in Cell Lines/Tissues", "Contamination of Cell Lines/Tissues",
        "Contamination of Materials (General)", "Contamination of Materials",
        "Contamination of Reagents", "Manipulation of Results", "Manipulation of Data",
        "Unreliable Image", "Error in Image", "Sabotage of Materials/Methods",
    ],
    "Computer-Generated Content and AI": [
        "Computer-Aided Content or Computer-Generated Content",
    ],
    "Authorship and Ethical Concerns": [
        "Concerns/Issues About Authorship", "Concerns/Issues about Animal Welfare",
        "Concerns/Issues about Human Subject Welfare", "Concerns/Issues About Authorship/Affiliation",
        "Misconduct by Author", "Ethical Violations by Author",
        "Ethical Violations by Company/Institution/Third Party", "False/Forged Authorship",
        "False/Forged Affiliation", "Conflict of Interest", "Lack of Approval from Author",
        "Lack of Approval from Company/Institution", "Lack of Approval from Third Party",
        "Lack of IRB/IACUC Approval and/or Compliance", "Informed/Patient Consent - None/Withdrawn",
        "Author Unresponsive", "False Affiliation",
    ],
    "Plagiarism and Duplication": [
        "Duplication of Image", "Duplication of Article", "Duplication of Data",
        "Duplication of Text", "Plagiarism of Article", "Euphemisms for Plagiarism",
        "Plagiarism of Text", "Plagiarism of Image", "Plagiarism of Data",
        "Euphemisms for Duplication", "Duplication of/in Article", "Duplication of/in Image",
        "Plagiarism of/in Article",
    ],
    "Image Manipulation and Fabrication": [
        "Manipulation of Images", "Concerns/Issues About Image", "Falsification/Fabrication of Image",
    ],
    "Investigations and Findings": [
        "Investigation by Journal/Publisher", "Investigation by Company/Institution",
        "Investigation by Third Party", "Investigation by ORI",
        "Misconduct - Official Investigation/Finding", "Investigation by Office of Research Integrity",
    ],
    "Peer Review and Editorial Issues": [
        "Fake Peer Review", "Rogue Editor", "Concerns/Issues with Peer Review",
        "Taken via Peer Review", "Compromised Peer Review", "Concerns/Issues about Peer Review",
    ],
    "Misconduct and Fraud": [
        "Paper Mill", "Falsification/Fabrication of Data", "Falsification/Fabrication of Results",
        "Misconduct by Third Party", "Misconduct by Company/Institution",
        "Misconduct - Official Investigation/Finding", "Euphemisms for Misconduct",
        "Randomly Generated Content", "Ethical Violations by Third Party",
        "Misconduct - Official Investigation(s) and/or Finding(s)",
    ],
    "Procedural and Legal Issues": [
        "Legal Reasons/Legal Threats", "Legal Reasons and/or Threats", "Criminal Proceedings",
        "Civil Proceedings", "Nonpayment of Fees/Refusal to Pay", "Publishing Ban",
    ],
    "Withdrawal and Retraction Notices": [
        "Withdrawal", "Retract and Replace", "Temporary Removal",
        "Date of Retraction/Other Unknown", "Notice - Limited or No Information",
        "Notice - Lack of", "Notice - Unable to Access via current resources",
        "Upgrade/Update of Prior Notice", "Updated to Retraction",
        "Notice - No/Limited Information", "Updated to Expression of Concern",
        "Upgrade/Update of Prior Notice(s)",
    ],
    "Complaints and Objections": [
        "Objections by Third Party", "Objections by Author(s)",
        "Objections by Company/Institution", "Complaints about Author",
        "Complaints about Third Party", "Complaints about Company/Institution",
    ],
    "Institutional and Policy Issues": [
        "Breach of Policy by Author", "Lack of IRB/IACUC Approval",
        "Concerns/Issues about Third Party Involvement",
        "Concerns/Issues about Referencing/Attributions", "Cites Retracted Work",
    ],
    "Miscellaneous": [
        "Copyright Claims", "Transfer of Copyright and/or Ownership",
        "Error by Journal/Publisher", "Error by Third Party",
        "Date of Article and/or Notice Unknown",
        "Duplicate Publication through Error by Journal/Publisher",
        "Duplication of Content through Error by Journal/Publisher",
        "Bias Issues or Lack of Balance", "Doing the Right Thing", "Salami Slicing",
        "Miscommunication by Author", "Miscommunication by Journal/Publisher",
        "Miscommunication by Company/Institution", "Miscommunication by Third Party",
        "Miscommunication with/by Author", "Miscommunication with/by Company/Institution",
        "Miscommunication with/by Journal/Publisher", "Miscommunication with/by Third Party",
        "Taken from Dissertation/Thesis", "Taken via Translation", "Not Presented at Conference",
        "Withdrawn to Publish in Different Journal", "Sabotage of Materials (Miscellaneous)",
        "EOC Lifted", "Hoax Paper", "No Further Action",
        "Nonpayment of Fees and/or Refusal to Pay",
        "Original Data and/or Images not Provided and/or not Available",
        "Withdrawn as Out of Date", "Removed", "Updated to Correction",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
#  CITATION ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

# Contact email required by the OpenAlex and NCBI polite pools. Replace this.
CONTACT_EMAIL = "your-email@example.com"

# Cap the number of papers sent to each citation API (set high to fetch all).
CITATION_SAMPLE_SIZE = 10_000

# Cache the enriched dataframe here so repeated runs don't re-hit the APIs.
CITATION_CACHE_PATH = DATA_DIR / "citation_cache.parquet"

# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICAL / ANALYSIS PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Country-level normalisation: minimum total publications to be a "major nation".
MAJOR_NATION_MIN_DOCS = 100_000
# Minimum papers for a country to appear in the temporal-lag analysis.
MIN_PAPERS_PER_COUNTRY = 50
# Time-to-retraction values outside this range (years) are treated as artifacts.
TTR_MIN_YEARS = 0.0
TTR_MAX_YEARS = 30.0

# Post-retraction contamination guardrails (Stage 1 fix):
# require a meaningful pre-retraction window and exclude very recent retractions
# so the post-retraction window is long enough to be meaningful.
CONTAMINATION_MIN_PRE_YEARS   = 2.0    # paper must have been out >= 2 yr before retraction
CONTAMINATION_MAX_RETRACT_YEAR = 2023  # exclude retractions after this year
CONTAMINATION_CURRENT_YEAR     = 2026  # used to size the post window

# Bootstrap iterations for confidence intervals.
BOOTSTRAP_N = 5000
# Global random seed for reproducibility.
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE STYLE
# ─────────────────────────────────────────────────────────────────────────────

# Colourblind-friendly, print-safe 13-colour palette.
PALETTE = [
    "#264653", "#2A9D8F", "#E9C46A", "#F4A261",
    "#E76F51", "#457B9D", "#A8DADC", "#6D6875",
    "#B5838D", "#606C38", "#DDA15E", "#BC6C25",
    "#8D99AE",
]

FIG_DPI  = 300
