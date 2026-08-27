#!/usr/bin/env python3
"""Synchronize the audited final manuscript table fragments.

Granular and intermediate table products remain under results/round4/generated.
This step ensures that results/derived always exposes the exact fragments used
by the current manuscript, rather than stale pre-consolidation versions.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "manuscript_tables"
DEST = ROOT / "results" / "derived"
DEST.mkdir(parents=True, exist_ok=True)
files = sorted(SOURCE.glob("*.tex"))
if not files:
    raise SystemExit("No final manuscript table fragments found")
for src in files:
    shutil.copy2(src, DEST / src.name)
print(f"Synchronized {len(files)} final manuscript tables into {DEST}")

# Refresh the result-file checksum manifest after replacing manuscript-facing
# fragments, so the released manifest describes the final repository state.
import csv
import hashlib
manifest = DEST / "release_file_manifest_sha256.csv"
rows = []
for path in sorted((ROOT / "results").rglob("*")):
    if path.is_file() and path != manifest:
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
with manifest.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Refreshed {manifest} with {len(rows)} entries")
