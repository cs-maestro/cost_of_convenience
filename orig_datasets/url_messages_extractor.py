#!/usr/bin/env python3
"""
url_messages_extractor.py

Reads 'merged_databases.csv', finds rows whose 'message' contains URLs,
cleans each URL (strips trailing punctuation, removes non‑ASCII tails, drops
everything from the first comma onward, strips unicode marks, skips URLs with '*'),
validates it, and writes a CSV of unique, valid URLs with their to_number,
from_number, time, and full message.

Usage:
  pip install validators
  python url_messages_extractor.py \
      --input merged_databases.csv \
      --output unique_url_messages.csv
"""
import csv
import re
import argparse
import validators

def main():
    parser = argparse.ArgumentParser(
        description="Extract unique, valid URLs from messages in a large CSV."
    )
    parser.add_argument(
        '--input', default='merged_databases.csv',
        help='Path to the input CSV file containing message rows'
    )
    parser.add_argument(
        '--output', default='unique_url_messages.csv',
        help='Path to the output CSV file for unique URL rows'
    )
    args = parser.parse_args()

    # 1) Rough URL matcher: http(s):// or www. up to whitespace or common delimiters
    url_pattern = re.compile(
        r'(?:https?://|www\.)'          # protocol or www.
        r'[^\s"\'<>()\[\]{}]+'          # URL body
        r'(?=[\s"\'<>()\[\]{}]|$)'      # stop at whitespace, quotes, brackets, or EOS
    )
    # 2) Strip ASCII trailing punctuation
    strip_trail = re.compile(r'[.,;:!?]+$')
    # 3) Keep only ASCII prefix
    ascii_prefix = re.compile(r'^[\x00-\x7F]+')

    seen_urls = set()
    total_with_url = 0
    unique_count = 0

    with open(args.input, 'r', newline='', encoding='utf-8') as in_f, \
         open(args.output, 'w', newline='', encoding='utf-8') as out_f:
        reader = csv.DictReader(in_f)
        fieldnames = ['url', 'to_number', 'from_number', 'time', 'message']
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            message = row.get('message', '')
            raw_matches = url_pattern.findall(message)
            if not raw_matches:
                continue
            total_with_url += 1

            for raw_url in raw_matches:
                url = raw_url

                # a) Drop anything from the first ASCII or Chinese comma onward
                for delim in [',', '，']:
                    if delim in url:
                        url = url.split(delim, 1)[0]

                # b) Strip any trailing ASCII punctuation
                url = strip_trail.sub('', url)

                # c) Remove directional marks U+200E/U+200F
                url = url.replace('\u200e', '').replace('\u200f', '')

                # d) Keep only the ASCII prefix
                m = ascii_prefix.match(url)
                if m:
                    url = m.group(0)
                else:
                    # no valid ASCII content, skip
                    continue

                # e) Skip URLs containing '*'
                if '*' in url:
                    continue

                # f) Skip if too short or malformed
                to_validate = url if url.startswith(('http://', 'https://')) else 'http://' + url
                if not validators.url(to_validate):
                    continue

                # g) Dedupe and write
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                unique_count += 1

                writer.writerow({
                    'url': url,
                    'to_number': row.get('to_number', ''),
                    'from_number': row.get('from_number', ''),
                    'time': row.get('time', ''),
                    'message': message
                })

    print(f"Total rows with at least one URL: {total_with_url}")
    print(f"Total unique, valid URLs extracted: {unique_count}")

if __name__ == '__main__':
    main()

