.PHONY: help install all clean prep descriptive subjects citations \
        citation-analysis stats networks

PYTHON ?= python

help:
	@echo "Targets:"
	@echo "  make install           Install Python dependencies"
	@echo "  make all               Run the full pipeline (same as run_all.sh)"
	@echo "  make prep              Stage 1: data preparation"
	@echo "  make descriptive       Stage 2: descriptive analyses"
	@echo "  make subjects          Stage 3: per-subject analyses"
	@echo "  make citations         Stage 4: citation enrichment (API calls)"
	@echo "  make citation-analysis Stage 5: citation analyses"
	@echo "  make stats             Stage 6: statistical modelling"
	@echo "  make networks          Stage 7: network analyses"
	@echo "  make clean             Remove generated figures, tables, and caches"

install:
	$(PYTHON) -m pip install -r requirements.txt

all:
	bash run_all.sh

prep:
	$(PYTHON) src/data_prep.py

descriptive: prep
	$(PYTHON) src/descriptive.py

subjects: prep
	$(PYTHON) src/subject_analysis.py

citations: prep
	$(PYTHON) src/citation_enrichment.py

citation-analysis: citations
	$(PYTHON) src/citation_analysis.py

stats: prep
	$(PYTHON) src/statistics_models.py

networks: prep
	$(PYTHON) src/network_analysis.py

clean:
	rm -f figures/*.pdf figures/*.png figures/*.html
	rm -f tables/*.csv
	rm -f data/clean_retractions.parquet data/enriched_retractions.parquet data/citation_cache.parquet
	@echo "Cleaned generated outputs and caches."
