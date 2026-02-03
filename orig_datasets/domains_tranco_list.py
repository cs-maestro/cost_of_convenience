import csv
from urllib.parse import urlparse

INPUT_CSV = "domain_analysis.csv"
TRANCO_CSV = "../tranco_2_2_2026.csv"
OUTPUT_CSV = "tranco_domain_analysis.csv"

COLUMN_INDEX = 1  # column 2 (0-based index)

def normalize_domain(value):
    value = value.strip()
    if not value:
        return None
    if "://" not in value:
        value = "http://" + value
    parsed = urlparse(value)
    return parsed.netloc.lower()

# Load Tranco ranks into a dict
tranco_rank = {}
with open(TRANCO_CSV, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    for rank, domain in reader:
        tranco_rank[domain.lower()] = int(rank)

# Read input CSV and collect unique domains
unique_domains = set()
rows = []

with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        domain = normalize_domain(row[COLUMN_INDEX])
        unique_domains.add(domain)
        rows.append(row + [domain])

# Map domains to Tranco rank
domain_to_rank = {
    d: tranco_rank.get(d) for d in unique_domains if d
}

# Write output CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header + ["normalized_domain", "tranco_rank"])
    for row in rows:
        domain = row[-1]
        writer.writerow(row + [domain_to_rank.get(domain)])

print(f"Done. Wrote {OUTPUT_CSV}")
