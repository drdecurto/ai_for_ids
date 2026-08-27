PYTHON ?= python

.PHONY: all analysis round4 sync-final verify audit test manifest clean-derived

all: analysis round4 verify audit sync-final test

analysis:
	$(PYTHON) scripts/recompute_manuscript_results.py

round4:
	$(PYTHON) scripts/recompute_round4_revision.py

sync-final:
	$(PYTHON) scripts/sync_final_manuscript_tables.py

verify:
	$(PYTHON) scripts/verify_manuscript_claims.py

audit:
	$(PYTHON) scripts/audit_manuscript_tables.py

test:
	$(PYTHON) -m pytest -q tests/test_release.py tests/test_round4_revision.py

manifest: analysis round4 audit sync-final
	@echo "Manifests regenerated under results/derived and results/round4/derived"

clean-derived:
	find results/derived -maxdepth 1 -type f -delete
	find results/round4/derived -maxdepth 1 -type f -delete
	find results/round4/generated -maxdepth 1 -type f -delete
