"""
build_data.py
=============
Produce the per-paper dataset the interactive dashboard reads
(docs/data.json), from the Retraction Watch CSV.

Steps:
  1. Keep only Retraction notices in the four biomedical subject areas.
  2. Map the ~90 raw reasons to 13 categories; compute time-to-retraction,
     author count, primary country, etc.
  3. Enrich each paper with a citation count and NIH Relative Citation Ratio
     from iCite (by PubMed ID) and OpenAlex (by DOI, BATCHED and correctly
     URL-encoded), plus per-year citation counts so the dashboard can show
     pre/post-retraction citation rates.
  4. Write a compact JSON of per-paper records.

The dashboard does ALL filtering and charting in the browser from this file, so
readers interact with the data without uploading anything.

Usage:
    python build_data.py path/to/retraction_watch.csv docs/data.json
    python build_data.py path/to/retraction_watch.csv docs/data.json --no-citations

Note on OpenAlex: DOIs must be passed RAW (lowercased, no https:// prefix) in a
piped filter and encoded exactly once by the HTTP layer. The earlier version
pre-encoded them and then let requests encode again, double-encoding '/' into
'%252F' so nothing matched. This version fixes that.
"""

import sys, json, math, time
from datetime import datetime, date
import pandas as pd
import numpy as np

# ---- config ----
SUBJECT_SCOPE = {
    "(bls) neuroscience": "Neuroscience",
    "(hsc) biostatistics/epidemiology": "Biostatistics/Epidemiology",
    "(hsc) radiology/imaging": "Radiology/Imaging",
    "(phy) nanotechnology": "Nanotechnology",
}
CONTACT_EMAIL = "your-email@example.com"   # used for the OpenAlex/NCBI polite pool
CURRENT_YEAR = date.today().year

REASON_CATEGORIES = {
  "Data and Results Issues":["unreliable results","unreliable results and/or conclusions","concerns/issues about data","error in data","original data not provided","unreliable data","error in results and/or conclusions","error in analyses","error in methods","results not reproducible","concerns/issues about results","concerns/issues about results and/or conclusions","concerns/issues about article","concerns/issues about methods","error in text","error in materials (general)","error in materials","error in cell lines/tissues","contamination of cell lines/tissues","contamination of materials (general)","contamination of materials","contamination of reagents","manipulation of results","manipulation of data","unreliable image","error in image","sabotage of materials/methods"],
  "Computer-Generated Content and AI":["computer-aided content or computer-generated content"],
  "Authorship and Ethical Concerns":["concerns/issues about authorship","concerns/issues about animal welfare","concerns/issues about human subject welfare","concerns/issues about authorship/affiliation","misconduct by author","ethical violations by author","ethical violations by company/institution/third party","false/forged authorship","false/forged affiliation","conflict of interest","lack of approval from author","lack of approval from company/institution","lack of approval from third party","lack of irb/iacuc approval and/or compliance","informed/patient consent - none/withdrawn","author unresponsive","false affiliation"],
  "Plagiarism and Duplication":["duplication of image","duplication of article","duplication of data","duplication of text","plagiarism of article","euphemisms for plagiarism","plagiarism of text","plagiarism of image","plagiarism of data","euphemisms for duplication","duplication of/in article","duplication of/in image","plagiarism of/in article"],
  "Image Manipulation and Fabrication":["manipulation of images","concerns/issues about image","falsification/fabrication of image"],
  "Investigations and Findings":["investigation by journal/publisher","investigation by company/institution","investigation by third party","investigation by ori","misconduct - official investigation/finding","investigation by office of research integrity"],
  "Peer Review and Editorial Issues":["fake peer review","rogue editor","concerns/issues with peer review","taken via peer review","compromised peer review","concerns/issues about peer review"],
  "Misconduct and Fraud":["paper mill","falsification/fabrication of data","falsification/fabrication of results","misconduct by third party","misconduct by company/institution","euphemisms for misconduct","randomly generated content","ethical violations by third party","misconduct - official investigation(s) and/or finding(s)"],
  "Procedural and Legal Issues":["legal reasons/legal threats","legal reasons and/or threats","criminal proceedings","civil proceedings","nonpayment of fees/refusal to pay","publishing ban"],
  "Withdrawal and Retraction Notices":["withdrawal","retract and replace","temporary removal","date of retraction/other unknown","notice - limited or no information","notice - lack of","notice - unable to access via current resources","upgrade/update of prior notice","updated to retraction","notice - no/limited information","updated to expression of concern","upgrade/update of prior notice(s)"],
  "Complaints and Objections":["objections by third party","objections by author(s)","objections by company/institution","complaints about author","complaints about third party","complaints about company/institution"],
  "Institutional and Policy Issues":["breach of policy by author","lack of irb/iacuc approval","concerns/issues about third party involvement","concerns/issues about referencing/attributions","cites retracted work"],
  "Miscellaneous":["copyright claims","transfer of copyright and/or ownership","error by journal/publisher","error by third party","salami slicing","hoax paper","no further action","removed","updated to correction"],
}
REASON_LOOKUP = {m: c for c, ms in REASON_CATEGORIES.items() for m in ms}


def map_reasons(raw):
    if pd.isna(raw) or not str(raw).strip():
        return ["Other / unknown"]
    cats = set()
    for tok in str(raw).split(";"):
        tok = tok.strip().lower()
        if tok:
            cats.add(REASON_LOOKUP.get(tok, "Other / unknown"))
    return sorted(cats) if cats else ["Other / unknown"]


def detect_subject(subj):
    if pd.isna(subj):
        return None
    low = str(subj).lower()
    for tag, name in SUBJECT_SCOPE.items():
        if tag in low:
            return name
    return None


# ---------------------------------------------------------------------------
#  CITATION ENRICHMENT
# ---------------------------------------------------------------------------

def fetch_icite(pmids):
    import requests
    out = {}
    for i in range(0, len(pmids), 100):
        chunk = ",".join(str(int(x)) for x in pmids[i:i + 100])
        try:
            r = requests.get("https://icite.od.nih.gov/api/pubs",
                             params={"pmids": chunk,
                                     "fields": "pmid,citation_count,relative_citation_ratio"},
                             timeout=30)
            r.raise_for_status()
            for rec in r.json().get("data", []):
                out[int(rec["pmid"])] = {
                    "cites": rec.get("citation_count"),
                    "rcr": rec.get("relative_citation_ratio"),
                }
        except Exception as e:
            print(f"    iCite chunk {i} failed: {e}")
        time.sleep(0.1)
    return out


def fetch_openalex(dois):
    """Batched DOI lookup. DOIs are passed RAW and encoded once by requests."""
    import requests
    out = {}
    CHUNK = 40
    headers = {"User-Agent": f"retraction-analysis (mailto:{CONTACT_EMAIL})"}
    for i in range(0, len(dois), CHUNK):
        chunk = [str(d).lower().strip() for d in dois[i:i + CHUNK] if str(d).strip()]
        if not chunk:
            continue
        # key once, values piped, RAW dois -> requests encodes exactly once
        filt = "doi:" + "|".join(chunk)
        for attempt in range(4):
            try:
                r = requests.get("https://api.openalex.org/works",
                                 params={"filter": filt, "per-page": CHUNK,
                                         "select": "doi,cited_by_count,counts_by_year",
                                         "mailto": CONTACT_EMAIL},
                                 headers=headers, timeout=45)
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if r.status_code != 200:
                    if i == 0:
                        print(f"    OpenAlex returned {r.status_code}: {r.text[:150]}")
                    break
                for w in r.json().get("results", []):
                    raw = (w.get("doi") or "").replace("https://doi.org/", "").lower()
                    if raw:
                        out[raw] = {
                            "cites": w.get("cited_by_count"),
                            "by_year": {str(c["year"]): c["cited_by_count"]
                                        for c in (w.get("counts_by_year") or [])},
                        }
                break
            except Exception as e:
                if i == 0:
                    print(f"    OpenAlex batch failed: {e}")
                break
        time.sleep(0.15)
    return out


def enrich(df):
    print("  Enriching citations (iCite + OpenAlex)...")
    try:
        import requests  # noqa: F401
    except ImportError:
        print("  requests not installed; skipping citations.")
        return df

    df["cites"] = np.nan
    df["rcr"] = np.nan
    df["cites_by_year"] = [dict() for _ in range(len(df))]

    # iCite by PMID
    pmids = df["OriginalPaperPubMedID"].dropna().astype(int)
    pmids = pmids[pmids > 0].unique()
    print(f"  iCite: {len(pmids):,} PMIDs")
    icite = fetch_icite(pmids)
    print(f"  iCite matched: {len(icite):,}")
    pmid_col = df["OriginalPaperPubMedID"]
    for idx in df.index:
        p = pmid_col[idx]
        if pd.notna(p) and int(p) in icite:
            df.at[idx, "cites"] = icite[int(p)]["cites"]
            df.at[idx, "rcr"] = icite[int(p)]["rcr"]

    # OpenAlex by DOI for the rest (also gives per-year counts for everyone we can)
    need = df[df["OriginalPaperDOI"].notna()]
    dois = need["OriginalPaperDOI"].dropna().unique().tolist()
    print(f"  OpenAlex: {len(dois):,} DOIs")
    oa = fetch_openalex(dois)
    print(f"  OpenAlex matched: {len(oa):,}")
    for idx in df.index:
        d = df.at[idx, "OriginalPaperDOI"]
        if pd.isna(d):
            continue
        key = str(d).lower().strip()
        if key in oa:
            if pd.isna(df.at[idx, "cites"]) and oa[key]["cites"] is not None:
                df.at[idx, "cites"] = oa[key]["cites"]
            df.at[idx, "cites_by_year"] = oa[key]["by_year"]

    matched = df["cites"].notna().sum()
    print(f"  Total citation-matched: {matched:,} / {len(df):,} ({matched/len(df)*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/retraction_watch.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "docs/data.json"
    do_citations = "--no-citations" not in sys.argv

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["RetractionNature"].astype(str).str.strip() == "Retraction"].copy()
    df["subject"] = df["Subject"].apply(detect_subject)
    df = df[df["subject"].notna()].copy()
    print(f"Biomedical retractions: {len(df):,}")

    pub = pd.to_datetime(df["OriginalPaperDate"], errors="coerce")
    ret = pd.to_datetime(df["RetractionDate"], errors="coerce")
    df["pub_year"] = pub.dt.year
    df["ret_year"] = ret.dt.year
    ttr = (ret - pub).dt.days / 365.25
    df["ttr"] = [round(float(t), 2) if (pd.notna(t) and 0 <= t <= 30) else None for t in ttr]
    df["reasons"] = df["Reason"].apply(map_reasons)
    df["countries"] = df["Country"].apply(
        lambda c: [x.strip() for x in str(c).split(";") if x.strip()] if pd.notna(c) else []
    )
    df["author_count"] = df["Author"].apply(
        lambda a: len([x for x in str(a).split(";") if x.strip()]) if pd.notna(a) else None
    )
    df["paywalled"] = df["Paywalled"].astype(str).str.strip().str.upper().map(
        {"YES": True, "NO": False}
    )

    if do_citations:
        df = enrich(df)
    else:
        df["cites"] = np.nan
        df["rcr"] = np.nan
        df["cites_by_year"] = [dict() for _ in range(len(df))]

    # Build compact per-paper records (drop 2026+ / future years handled in UI)
    records = []
    for _, r in df.iterrows():
        cites = r.get("cites")
        cby = r.get("cites_by_year") or {}
        # pre / post retraction citation split (needs by_year + a valid ret_year)
        pre = post = None
        if cby and pd.notna(r["ret_year"]):
            ry = int(r["ret_year"])
            pre = sum(v for y, v in cby.items() if int(y) < ry)
            post = sum(v for y, v in cby.items() if int(y) >= ry)
        records.append({
            "sub": r["subject"],
            "py": int(r["pub_year"]) if pd.notna(r["pub_year"]) else None,
            "ry": int(r["ret_year"]) if pd.notna(r["ret_year"]) else None,
            "ttr": r["ttr"],
            "rs": r["reasons"],
            "co": r["countries"],
            "jr": (str(r["Journal"]).strip() if pd.notna(r["Journal"]) else ""),
            "pb": (str(r["Publisher"]).strip() if pd.notna(r["Publisher"]) else ""),
            "ac": int(r["author_count"]) if pd.notna(r["author_count"]) else None,
            "ct": int(cites) if pd.notna(cites) else None,
            "rcr": round(float(r["rcr"]), 2) if pd.notna(r.get("rcr")) else None,
            "pre": int(pre) if pre is not None else None,
            "post": int(post) if post is not None else None,
            "pw": bool(r["paywalled"]) if pd.notna(r.get("paywalled")) else None,
        })

    out = {
        "generated": datetime.utcnow().strftime("%Y-%m-%d"),
        "current_year": CURRENT_YEAR,
        "subjects": list(SUBJECT_SCOPE.values()),
        "n": len(records),
        "has_citations": bool(df["cites"].notna().any()),
        "records": records,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Wrote {out_path}  ({len(json.dumps(out))/1024:.0f} KB, {len(records):,} papers, "
          f"citations={'yes' if out['has_citations'] else 'no'})")


if __name__ == "__main__":
    main()
