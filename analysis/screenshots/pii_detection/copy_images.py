#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv, shutil
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────────────────
CSV_PATH   = Path("pii_results.csv")               # your CSV file
IMAGES_DIR = Path("s_dedupe/unique_ss")            # folder where original images are
PII_DIR    = Path("pii")                           # destination for "Y"
NO_PII_DIR = Path("no_pii")                        # destination for "N"

# Create output dirs
PII_DIR.mkdir(parents=True, exist_ok=True)
NO_PII_DIR.mkdir(parents=True, exist_ok=True)

# ─── Process CSV ─────────────────────────────────────────────────────────────
with CSV_PATH.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_name = row["image_name"].strip()
        result   = row["result"].strip()

        src = IMAGES_DIR / img_name
        if not src.exists():
            print(f"[!] Missing image: {src}")
            continue

        if result.startswith("Y"):
            dst = PII_DIR / img_name
        else:
            dst = NO_PII_DIR / img_name

        shutil.copy2(src, dst)
        print(f"Copied {img_name} → {dst.parent.name}")

print("\nDone! Images split into:")
print(f" - {PII_DIR.resolve()}")
print(f" - {NO_PII_DIR.resolve()}")
