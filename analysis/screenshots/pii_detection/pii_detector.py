#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PII detector using LM Studio (OpenAI-compatible API) with robust logging
and automatic fuzzy fallback to the closest PII_TYPES label(s).
"""

import csv
import json
import os
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple, Dict, Any

from tqdm import tqdm

# ─── OpenAI-compatible client (LM Studio) ─────────────────────────────────────
try:
    from openai import OpenAI
except Exception as e:
    print("Please install the OpenAI SDK:  pip install openai", file=sys.stderr)
    raise

# ─── LM Studio Connection ─────────────────────────────────────────────────────
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LMSTUDIO_API_KEY  = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
LMSTUDIO_MODEL    = os.environ.get("LMSTUDIO_MODEL", "openai/gpt-oss-120b")

client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)

# ─── Config ───────────────────────────────────────────────────────────────────
OCR_CSV_PATH   = Path("../ocr/ocr.csv")
CSV_OUT_PATH   = Path("pii_results.csv")
JSONL_OUT_PATH = Path("pii_results.jsonl")

MAX_OCR_CHARS  = 25_000
FLUSH_EVERY_N  = 10

# Generation settings
MAX_TOKENS   = 1024
TEMPERATURE  = 0.0
TOP_P        = 1.0

# ─── PII Types (UPDATED) ──────────────────────────────────────────────────────
PII_TYPES = [
    "user's mothers maiden name",
    "user's password",
    "user's security question",
    "user's user name",
    "user's products purchased",
    "user's psychological trends",
    "user's search history",
    "user's services purchased",
    "user's email address",
    "user's email content",
    "user's postal address",
    "user's telephone number",
    "user's text messages",
    "user's citizenship",
    "user's color",
    "user's date of birth",
    "user's education",
    "user's employment history",
    "user's family",
    "user's first name",
    "user's gender",
    "user's gender identity",
    "user's hair color",
    "user's height",
    "user's language",
    "user's marital status",
    "user's military or veteran status",
    "user's place of birth",
    "user's race",
    "user's religion",
    "user's sex life",
    "user's union membership",
    "user's immigration status",
    "user's last name",
    "user's bank account number",
    "user's credit card number",
    "user's debit card number",
    "user's financial account number",
    "user's insurance policy number",
    "user's driver authorization card number",
    "user's health insurance identification number",
    "user's individual taxpayer identification number",
    "user's medical identification number",
    "user's military identification card number",
    "user's nondriver state identification card number",
    "user's passport number",
    "user's social security number",
    "user's state identification card number",
    "user's drivers license number",
    "user's aids",
    "user's breastfeeding",
    "user's cancer",
    "user's childbirth",
    "user's chiropractic",
    "user's diagnosis",
    "user's disability",
    "user's health records",
    "user's medicine",
    "user's mental condition",
    "user's pregnancy",
    "user's genetic data",
    "user's hiv",
    "user's request for family care leave",
    "user's request for leave for an employees own serious health condition",
    "user's location",
    "user's records of personal property"
]
PII_SET = {p.lower() for p in PII_TYPES}
CANONICALS: Dict[str, str] = {p.lower(): p for p in PII_TYPES}

# ─── Prompt ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a PII detection assistant.\n"
    "Reasoning: high\n"
    "ALWAYS answer in exactly the format below:\n"
    "Y,{comma_separated_PII_TYPES}  OR  N,{}\n"
    "NEVER include explanations or extra text.\n"
    "Use only labels exactly as listed in PII_TYPES (verbatim strings).\n"
    "Only flag the USER PII, not website or company information."
)

def build_user_prompt(ocr_text: str) -> str:
    ocr_text = (ocr_text or "").strip()
    if len(ocr_text) > MAX_OCR_CHARS:
        ocr_text = ocr_text[:MAX_OCR_CHARS]

    return (
        f"PII_TYPES = {PII_TYPES}\n"
        f"Does the OCR text below from a website screenshot contain any of the user PII from PII_TYPES and if so, which ones.\n"
        f"Give me results as Y,{{comma separated PII_TYPES}} for detected PII and N,{{}} for not detected PII.\n"
        f"Make sure to only flag the USER PII and not website or company information:\n\n"
        f"OCR:\n{ocr_text}\n\n"
        f"Answer:"
    )

# ─── Normalization helpers ───────────────────────────────────────────────────────
_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*")  # remove parenthetical segments
_YN_RE    = re.compile(r"^\s*([YyNn])\s*,\s*(.*)$")  # accepts with/without braces after comma

def _ascii_quotes(s: str) -> str:
    return (
        s.replace("\u2018", "'")
         .replace("\u2019", "'")
         .replace("\u201C", '"')
         .replace("\u201D", '"')
    )

def _simplify_label(s: str) -> str:
    """
    Lowercase, remove outer braces/quotes/parentheticals, collapse spaces.
    """
    if s is None:
        return ""
    s = _ascii_quotes(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()

    # strip outer braces if any, then remove any stray braces
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    s = s.replace("{", " ").replace("}", " ")

    # drop enclosing quotes
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        s = s[1:-1]

    # remove parentheticals
    s = _PAREN_RE.sub(" ", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()

# Build alias map for exact/simplified canonical matches
ALIASES: Dict[str, str] = {}
for p in PII_TYPES:
    ALIASES[p.lower()] = p
    ALIASES[_simplify_label(p)] = p

def _match_label(token: str) -> str | None:
    """
    Exact / simplified canonical match. Returns canonical label or None.
    """
    if not token:
        return None
    raw = _ascii_quotes(str(token)).strip().strip(",")
    if not raw:
        return None

    # 1) direct canonical (case-insensitive)
    t_lc = raw.lower()
    if t_lc in CANONICALS:
        return CANONICALS[t_lc]

    # 2) simplified (remove braces/quotes/parentheticals)
    t_simpl = _simplify_label(raw)
    if t_simpl in ALIASES:
        return ALIASES[t_simpl]

    return None

# ─── Automatic fuzzy fallback to closest PII_TYPES ────────────────────────────
def _normalize_for_similarity(s: str) -> str:
    """
    Simpler normalization for similarity: lowercase, ascii, strip punctuation to spaces.
    """
    s = _simplify_label(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _token_words(s: str) -> set:
    return set(_normalize_for_similarity(s).split())

def _word_jaccard(a: str, b: str) -> float:
    A, B = _token_words(a), _token_words(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

# Precompute simplified canonicals once
PII_SIMPLIFIED: Dict[str, str] = { _normalize_for_similarity(p): p for p in PII_TYPES }

def _sim_score(a: str, b: str) -> float:
    """
    Hybrid similarity: SequenceMatcher + word Jaccard + substring bonus.
    """
    a_n, b_n = _normalize_for_similarity(a), _normalize_for_similarity(b)
    r = SequenceMatcher(None, a_n, b_n).ratio()
    j = _word_jaccard(a_n, b_n)
    sub = 0.15 if (a_n in b_n or b_n in a_n) and min(len(a_n), len(b_n)) >= 3 else 0.0
    return 0.6 * r + 0.35 * j + sub

def _fuzzy_match_label(token: str, threshold: float = 0.72, epsilon: float = 0.02, max_hits: int = 2) -> List[str]:
    """
    Return up to max_hits best-matching canonical labels for 'token' if top score >= threshold.
    Includes ties within epsilon of the top score (avoid arbitrary tie-breaking).
    """
    if not token or not token.strip():
        return []
    scores: List[Tuple[float, str]] = []
    for canon_simpl, canon in PII_SIMPLIFIED.items():
        score = _sim_score(token, canon)
        scores.append((score, canon))
    scores.sort(key=lambda x: x[0], reverse=True)

    out: List[str] = []
    if scores and scores[0][0] >= threshold:
        top = scores[0][0]
        for sc, canon in scores:
            if sc + 1e-9 < top - epsilon:
                break
            if canon not in out:
                out.append(canon)
            if len(out) >= max_hits:
                break
    return out

def normalize_and_validate(answer_line: str):
    """
    Returns: (normalized_result_str, categories_list, parse_status)
      - Accepts: 'Y,{...}', 'Y, ...', 'N,{}', 'N, ...' (braces optional)
      - Exact/simplified match first; else automatic fuzzy fallback.
      - If still nothing matched but model said 'Y', fuzzy on the whole remainder.
      - parse_status ∈ {'parsed','no_match','empty'}
    """
    if answer_line is None:
        return "N,{}", [], "empty"

    original = str(answer_line)
    stripped = original.strip()
    if stripped == "":
        return "N,{}", [], "empty"

    m = _YN_RE.match(stripped)
    if not m:
        return "N,{}", [], "no_match"

    yn, rest = m.group(1).upper(), m.group(2)

    # Normalize remainder: allow "{a,b}" or just "a,b"
    rest_clean = rest.strip()
    if rest_clean.startswith("{") and rest_clean.endswith("}"):
        rest_clean = rest_clean[1:-1]

    # Remove stray braces and tidy spaces for token parsing
    rest_tokens_src = rest_clean.replace("{", " ").replace("}", " ")
    rest_tokens_src = re.sub(r"\s+", " ", rest_tokens_src).strip()

    # If explicit N, treat as no-PII regardless of contents
    if yn == "N":
        return "N,{}", [], "parsed"

    cats: List[str] = []
    seen = set()

    # 1) token-based matching (commas/semicolons), with fuzzy fallback per token
    if rest_tokens_src:
        parts = re.split(r"[;,]", rest_tokens_src)
        for raw_tok in (t for t in parts if t.strip()):
            canonical = _match_label(raw_tok)
            if canonical:
                if canonical not in seen:
                    seen.add(canonical)
                    cats.append(canonical)
                continue
            # fuzzy fallback for this token
            for cand in _fuzzy_match_label(raw_tok):
                if cand not in seen:
                    seen.add(cand)
                    cats.append(cand)

    # 2) If model said Y but nothing matched, attempt fuzzy on the whole remainder
    if not cats and rest_tokens_src:
        for cand in _fuzzy_match_label(rest_tokens_src, threshold=0.76, epsilon=0.02, max_hits=2):
            if cand not in seen:
                seen.add(cand)
                cats.append(cand)

    # Y but nothing mapped → conservative fallback to N,{}
    if not cats:
        return "N,{}", [], "parsed"

    return f"Y,{{{','.join(cats)}}}", cats, "parsed"

# ─── I/O ──────────────────────────────────────────────────────────────────────
def load_ocr_rows(csv_path: Path) -> List[Tuple[str, str]]:
    if not csv_path.is_file():
        raise SystemExit(f"Missing OCR CSV: {csv_path}")
    rows: List[Tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        need = {"image_name", "extracted_text"}
        if not need.issubset(set(reader.fieldnames or [])):
            raise SystemExit("OCR CSV must have headers: image_name, extracted_text")
        for row in reader:
            name = (row.get("image_name") or "").strip()
            text = row.get("extracted_text") or ""
            if name:
                rows.append((name, text))
    return rows

# ─── LM Studio Chat Call ──────────────────────────────────────────────────────
def lmstudio_complete(system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
    """
    Return (text, meta). On API error, returns ("", {"api_error": "..."}).
    'text' is the raw assistant message content (may be empty).
    meta includes finish_reason.
    """
    try:
        resp = client.chat.completions.create(
            model=LMSTUDIO_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        choice = resp.choices[0]
        text = choice.message.content if choice and choice.message else ""
        meta: Dict[str, Any] = {
            "finish_reason": getattr(choice, "finish_reason", None),
        }
        return text, meta
    except Exception as e:
        return "", {"api_error": f"{type(e).__name__}: {e}"}

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Connectivity sanity check
    try:
        _ = client.models.list()
    except Exception as e:
        print(
            f"[fatal] Could not reach LM Studio at {LMSTUDIO_BASE_URL}. "
            f"Is the server running? Error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    data = load_ocr_rows(OCR_CSV_PATH)
    if not data:
        print("[warn] No rows found in OCR CSV.")
        return

    with JSONL_OUT_PATH.open("w", encoding="utf-8") as jout, \
         CSV_OUT_PATH.open("w", encoding="utf-8", newline="") as fcsv:

        csv_writer = csv.DictWriter(fcsv, fieldnames=["image_name", "result"])
        csv_writer.writeheader()

        written = 0
        for img_name, text in tqdm(data, desc="Querying LM Studio"):
            user_prompt = build_user_prompt(text)
            raw_text, meta = lmstudio_complete(SYSTEM_PROMPT, user_prompt)

            # Keep both original (unstripped) and normalized (stripped)
            original_output = "" if raw_text is None else str(raw_text)
            stripped_output  = original_output.strip()

            result_str, cats, parse_status = normalize_and_validate(original_output)

            # JSONL (rich detail). We always log the original output, even if invalid/empty.
            record = {
                "image_name": img_name,
                "result": result_str,
                "categories": cats,
                "parse_status": parse_status,            # parsed | no_match | empty | api_error
                "model_output": stripped_output,         # stripped assistant text
                "original_output": original_output,      # exact bytes from the model
                "model_meta": meta,                      # finish_reason, usage, or api_error
            }
            if "api_error" in meta:
                record["parse_status"] = "api_error"

            jout.write(json.dumps(record, ensure_ascii=False) + "\n")
            csv_writer.writerow({"image_name": img_name, "result": result_str})

            written += 1
            if written % FLUSH_EVERY_N == 0:
                jout.flush()
                fcsv.flush()

    print(f"[done] Wrote {CSV_OUT_PATH} and {JSONL_OUT_PATH}")

if __name__ == "__main__":
    main()
