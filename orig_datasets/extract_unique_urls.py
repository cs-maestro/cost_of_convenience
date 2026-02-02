#!/usr/bin/env python3
"""
extract_unique_urls.py

Extract unique URLs from a CSV file
"""
import csv

infile  = 'unique_url_messages.csv'
outfile = 'unique_urls.txt'

unique_urls = set()
with open(infile, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        url = row.get('url','').strip()
        # simple sanity check
        if url.startswith(('http://','https://','www.')):
            unique_urls.add(url)

with open(outfile, 'w', encoding='utf-8') as f:
    for url in sorted(unique_urls):
        f.write(url + '\n')

print(f"Wrote {len(unique_urls)} unique URLs to {outfile}")