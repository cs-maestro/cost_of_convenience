# The Tragedy of Convenience: Research Artifact

This repository contains the research scripts and selected derived data for the
paper **"The Tragedy of Convenience: Cascading User-Data Leakage from SMS-delivered
URLs"**. The scripts cover SMS processing, URL collection, screenshot
deduplication, OCR, and classification of personally identifiable information
(PII) in screenshots and network responses.


The artifact requests ACM CCS **Artifacts Available** badge. The permanent public artifact 
archive is at https://doi.org/10.5281/zenodo.23157665. The public artifact provides source code, saved PII classification outputs,
PII vocabularies, and an aggregate domain summary. Raw SMS databases, message
text, phone numbers, SMS-delivered URL lists, URL-to-screenshot mappings,
screenshots, extracted OCR text, HAR files, and captured HTML/response bodies are
omitted because they can expose user data or access to private resources.
PII result files identify samples by hashed filenames and report 
detection labels and categories; the screenshot JSONL also retains model responses 
and metadata.

For repository questions, use the
[GitHub issue tracker](https://github.com/cs-maestro/cost_of_convenience/issues).
