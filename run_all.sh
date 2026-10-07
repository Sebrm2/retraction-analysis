#!/usr/bin/env bash
#
# run_all.sh
# ==========
# Runs the full biomedical retraction analysis pipeline end to end.
# Each stage is an independent Python module; stages communicate through
# cached parquet files in data/, so you can also run any stage on its own.
#
# Usage:
#   bash run_all.sh              # run every stage in order
#   bash run_all.sh --no-network # skip the (slower) network stage
#   bash run_all.sh --no-citations  # skip citation enrichment + citation analyses
#
set -euo pipefail

# Resolve the directory this script lives in, so it works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
RUN_NETWORK=1
RUN_CITATIONS=1

for arg in "$@"; do
  case "$arg" in
    --no-network)   RUN_NETWORK=0 ;;
    --no-citations) RUN_CITATIONS=0 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

echo "============================================================"
echo " Biomedical Retraction Analysis Pipeline"
echo "============================================================"

echo
echo ">>> Stage 1/7: Data preparation"
$PYTHON src/data_prep.py

echo
echo ">>> Stage 2/7: Descriptive analyses"
$PYTHON src/descriptive.py

echo
echo ">>> Stage 3/7: Per-subject analyses"
$PYTHON src/subject_analysis.py

if [ "$RUN_CITATIONS" -eq 1 ]; then
  echo
  echo ">>> Stage 4/7: Citation enrichment (network calls; may take a while)"
  $PYTHON src/citation_enrichment.py

  echo
  echo ">>> Stage 5/7: Citation analyses"
  $PYTHON src/citation_analysis.py
else
  echo
  echo ">>> Stages 4-5/7: Citation stages skipped (--no-citations)"
fi

echo
echo ">>> Stage 6/7: Statistical modelling"
$PYTHON src/statistics_models.py

if [ "$RUN_NETWORK" -eq 1 ]; then
  echo
  echo ">>> Stage 7/7: Network analyses"
  $PYTHON src/network_analysis.py
else
  echo
  echo ">>> Stage 7/7: Network stage skipped (--no-network)"
fi

echo
echo "============================================================"
echo " Done. Figures -> figures/   Tables -> tables/"
echo "============================================================"
