PYTHON ?= python

.PHONY: all analysis round4 round5 sync-final verify audit test manifest clean-derived

all: analysis round4 round5 verify audit sync-final test

analysis:
	$(PYTHON) scripts/recompute_manuscript_results.py

round4:
	$(PYTHON) scripts/recompute_round4_revision.py

round5:
	$(PYTHON) scripts/round5_record_analysis.py --results results --out results/round5/record
	$(PYTHON) scripts/integrate_mn5.py --summary results/round5/mn5 --out results/round5/mn5 --record results/round5/record
	$(PYTHON) scripts/make_round5_tables.py --record results/round5/record --mn5 results/round5/mn5 --crossovers results/derived/cost_crossovers_summary.csv --out results/round5/manuscript_tables


sync-final:
	$(PYTHON) scripts/sync_final_manuscript_tables.py

verify:
	$(PYTHON) scripts/verify_manuscript_claims.py

audit:
	$(PYTHON) scripts/audit_manuscript_tables.py

test:
	$(PYTHON) -m pytest -q tests/test_release.py tests/test_round4_revision.py tests/test_round5.py

manifest: analysis round4 audit sync-final
	@echo "Manifests regenerated under results/derived and results/round4/derived"

clean-derived:
	find results/derived -maxdepth 1 -type f -delete
	find results/round4/derived -maxdepth 1 -type f -delete
	find results/round4/generated -maxdepth 1 -type f -delete
