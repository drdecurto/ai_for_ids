PYTHON ?= python

.PHONY: all analysis verify audit test manifest clean-derived

all: analysis verify audit test

analysis:
	$(PYTHON) scripts/recompute_manuscript_results.py

verify:
	$(PYTHON) scripts/verify_manuscript_claims.py

audit:
	$(PYTHON) scripts/audit_manuscript_tables.py

test:
	$(PYTHON) -m pytest -q tests/test_release.py

manifest: analysis audit
	@echo "Manifest regenerated at results/derived/release_file_manifest_sha256.csv"

clean-derived:
	find results/derived -maxdepth 1 -type f -delete
