import csv
from urllib.parse import urlparse

INPUT_CSV = "combined_final_list.csv"
TRANCO_CSV = "tranco_2_2_2026.csv"
OUTPUT_CSV = "final_domains_with_tranco.csv"

COLUMN_INDEX = 3  # column 4 (0-based) - final domain

def normalize_domain(value):
    value = value.strip()
    if not value:
        return None
    if "://" not in value:
        value = "http://" + value
    parsed = urlparse(value)
    return parsed.netloc.lower()

# Load Tranco rankings
tranco_rank = {}
with open(TRANCO_CSV, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    for rank, domain in reader:
        tranco_rank[domain.lower()] = int(rank)

# Collect unique normalized domains
unique_domains = set()

with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader, None)  # skip header if present
    for row in reader:
        domain = normalize_domain(row[COLUMN_INDEX])
        if domain:
            unique_domains.add(domain)

# Write unique domains + ranks
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["domain", "tranco_rank"])
    for domain in sorted(unique_domains):
        writer.writerow([domain, tranco_rank.get(domain)])

print(f"Done. Wrote {len(unique_domains)} unique domains to {OUTPUT_CSV}")
