#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv, shutil
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────────────────
CSV_PATH   = Path("pii_results.csv")                          # your CSV file
IMAGES_DIR = Path("../../ss_dedupe/unique_ss")                # folder where original images are
PII_DIR    = Path("pii")                                      # destination for "Y"
NO_PII_DIR = Path("no_pii")                                   # destination for "N"

# Create output dirs
PII_DIR.mkdir(parents=True, exist_ok=True)
NO_PII_DIR.mkdir(parents=True, exist_ok=True)

# ─── Process CSV ─────────────────────────────────────────────────────────────
with CSV_PATH.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        log_name = row["log_name"].strip()
        if log_name.endswith(".txt"):
            log_name = log_name[:-4] + ".png"

        result   = row["result"].strip()

        src = IMAGES_DIR / log_name
        if not src.exists():
            print(f"[!] Missing image: {src}")
            continue

        if result.startswith("Y"):
            dst = PII_DIR / log_name
        else:
            dst = NO_PII_DIR / log_name

        shutil.copy2(src, dst)
        print(f"Copied {log_name} → {dst.parent.name}")

print("\nDone! Images split into:")
print(f" - {PII_DIR.resolve()}")
print(f" - {NO_PII_DIR.resolve()}")
