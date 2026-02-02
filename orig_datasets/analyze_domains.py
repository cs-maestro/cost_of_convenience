#!/usr/bin/env python3
"""
Outputs created in --outdir:
  - domain_analysis.csv          (per-URL mapping: url,domain)
  - domain_summary.csv           (Top-N + Others: domain,count,percentage)
  - domain_distribution.png      (bar chart of Top-N + Others by %)

"""

import argparse
import os
import sys
from urllib.parse import urlparse
from collections import Counter

# Optional dependency for accurate domain extraction
try:
    import tldextract  # type: ignore
    _HAS_TLDEXTRACT = True
except Exception:
    _HAS_TLDEXTRACT = False

import pandas as pd
import matplotlib.pyplot as plt


def _normalize_url(u: str) -> str:
    """Ensure URL has a scheme so urlparse works consistently."""
    u = u.strip()
    if not u:
        return ""
    if "://" not in u:
        # Treat bare domains/hosts as http by default
        return "http://" + u
    return u


def extract_domain(url: str) -> str:
    """Extract the registered domain from a URL."""
    if not url:
        return ""
    try:
        if _HAS_TLDEXTRACT:
            ext = tldextract.extract(url)
            # ext.domain can be empty for some edge cases (IPs, localhost, etc.)
            if ext.registered_domain:
                return ext.registered_domain.lower()
            # Fallback to netloc for non-standard cases
        # Fallback method (less accurate on multi-part TLDs)
        netloc = urlparse(url).netloc
        if not netloc:
            # If urlparse couldn't get netloc, try parsing path
            netloc = urlparse(_normalize_url(url)).netloc
        netloc = netloc.split("@")[-1]  # strip userinfo
        netloc = netloc.split(":")[0]   # strip port
        netloc = netloc.strip().lower()
        # If it's an IP or localhost, return as-is
        if netloc.replace(".", "").isdigit() or netloc in {"localhost", ""}:
            return netloc
        parts = [p for p in netloc.split(".") if p]
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return netloc
    except Exception:
        return ""


def load_urls(path: str) -> list[str]:
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            urls.append(_normalize_url(line))
    return urls


def make_per_url_df(urls: list[str]) -> pd.DataFrame:
    domains = [extract_domain(u) for u in urls]
    return pd.DataFrame({"url": urls, "domain": domains})


def make_summary(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    counts = Counter(df["domain"].fillna("").astype(str))
    # Remove empty domains if any
    if "" in counts:
        del counts[""]
    total = sum(counts.values())
    if total == 0:
        return pd.DataFrame(columns=["domain", "count", "percentage"])

    # Top-N
    top_items = counts.most_common(top_n)
    top_domains, top_counts = zip(*top_items) if top_items else ([], [])
    top_pct = [c * 100.0 / total for c in top_counts]

    # Others
    others_count = total - sum(top_counts) if top_items else 0
    rows = []
    for d, c, p in zip(top_domains, top_counts, top_pct):
        rows.append({"domain": d, "count": c, "percentage": round(p, 2)})
    if others_count > 0:
        rows.append({"domain": "Others", "count": others_count,
                     "percentage": round(others_count * 100.0 / total, 2)})

    return pd.DataFrame(rows)


def plot_top_bar(summary_df: pd.DataFrame, out_path: str, title: str = "Domain Distribution (Top 10 + Others)"):
    if summary_df.empty:
        # Create a tiny placeholder figure to avoid errors
        plt.figure(figsize=(6, 3))
        plt.title("No data to plot")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return

    # Bar chart of percentages
    x = summary_df["domain"].tolist()
    y = summary_df["percentage"].tolist()

    plt.figure(figsize=(12, 6))
    bars = plt.bar(x, y)
    plt.title(title)
    plt.xlabel("Domain")
    plt.ylabel("Percentage of URLs (%)")
    plt.xticks(rotation=45, ha="right")

    # Add % labels above bars
    for rect, pct in zip(bars, y):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height,
                 f"{pct:.2f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze domains from a list of URLs.")
    parser.add_argument("--input", "-i", default="unique_urls.txt", help="Path to unique_urls.txt")
    parser.add_argument("--outdir", "-o", default=".",help="Directory to write outputs")
    parser.add_argument("--top", "-n", type=int, default=10, help="How many top domains to plot/summarize (default: 10)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    try:
        urls = load_urls(args.input)
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not urls:
        print("No URLs found in input file.", file=sys.stderr)
        # Still create empty outputs for consistency
        empty_df = pd.DataFrame(columns=["url", "domain"])
        empty_df.to_csv(os.path.join(args.outdir, "domain_analysis.csv"), index=False)
        pd.DataFrame(columns=["domain", "count", "percentage"]).to_csv(
            os.path.join(args.outdir, "domain_summary.csv"), index=False
        )
        # Empty plot
        plot_top_bar(pd.DataFrame(), os.path.join(args.outdir, "domain_distribution.png"))
        sys.exit(0)

    per_url_df = make_per_url_df(urls)
    per_url_csv = os.path.join(args.outdir, "domain_analysis.csv")
    per_url_df.to_csv(per_url_csv, index=False)

    summary_df = make_summary(per_url_df, top_n=args.top)
    summary_csv = os.path.join(args.outdir, "domain_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    chart_path = os.path.join(args.outdir, "domain_distribution.png")
    plot_top_bar(summary_df, chart_path)

    # Console summary
    print(f"Wrote: {per_url_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {chart_path}")
    if not _HAS_TLDEXTRACT:
        print("Note: 'tldextract' not found; domain parsing used a fallback and may be less accurate on some TLDs.")


if __name__ == "__main__":
    main()
