#!/usr/bin/env python3
"""
merge_dbs_to_csv.py

Scans './database_files' for all '*.db' SQLite files,
reads every table in each database, and writes all rows
to 'merged_databases.csv' without exhausting memory.

Usage: python merge_dbs_to_csv.py [--db_dir DIR] [--output FILE] [--batch BATCH_SIZE]
"""
import os
import sqlite3
import csv
import argparse

def get_user_tables(conn):
    """Return non-system table names in the SQLite database."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    return [row[0] for row in cursor]


def merge_databases(db_dir, output_csv, batch_size):
    first_write = True
    with open(output_csv, 'w', newline='', encoding='utf-8') as out_f:
        writer = None

        for filename in sorted(os.listdir(db_dir)):
            if not filename.endswith('.db'):
                continue
            db_path = os.path.join(db_dir, filename)
            print(f"Processing {db_path}")
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row

            try:
                tables = get_user_tables(conn)
                for table in tables:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT * FROM `{table}`;")

                    # Initialize writer with header from first table of first DB
                    if first_write:
                        columns = [desc[0] for desc in cursor.description]
                        writer = csv.writer(out_f)
                        writer.writerow(columns)
                        first_write = False

                    # Stream rows in batches
                    while True:
                        rows = cursor.fetchmany(batch_size)
                        if not rows:
                            break
                        writer.writerows((tuple(r) for r in rows))
            finally:
                conn.close()

    print(f"Merge complete. Output written to '{output_csv}'")


def main():
    parser = argparse.ArgumentParser(
        description="Merge all SQLite .db files into a single CSV."
    )
    parser.add_argument(
        '--db_dir', default='./database_files',
        help='Directory containing .db files'
    )
    parser.add_argument(
        '--output', default='merged_databases.csv',
        help='Path to the output CSV file'
    )
    parser.add_argument(
        '--batch', type=int, default=100000,
        help='Number of rows to fetch per batch'
    )

    args = parser.parse_args()
    merge_databases(args.db_dir, args.output, args.batch)

if __name__ == '__main__':
    main()
