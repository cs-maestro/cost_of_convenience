#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator, Set
import json, re
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
LOGS_PATH      = Path("./logs")           # numbered JSON-only response files
CSV_OUT_PATH   = Path("pii_results.csv")
JSONL_OUT_PATH = Path("pii_results.jsonl")

# Generation settings
MAX_TOKENS   = 1024
TEMPERATURE  = 0.0
TOP_P        = 1.0

# Chunking long inputs
CHUNK_CHAR_LIMIT   = 50000
CHUNK_CHAR_OVERLAP = 50

# header: "1 [har:html] {...}"
_HEADER_RE = re.compile(
    r"(?m)^\s*(?P<num>\d+)\s*\[\s*(?P<origin>[^\]]+)\s*\]\s*"
)

# ─── PII Types ────────────────────────────────────────────────────────────────
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
    "Y,{label1 (example1),label2 (example2),...} OR N,{}\n"
    "For each flagged label, include exactly one example value in parentheses from the JSON snippet.\n"
    "NEVER include explanations or extra text.\n"
    "Use only labels exactly as listed in PII_TYPES (verbatim strings).\n"
    "Only flag the USER PII, not website or company information."
)

def build_user_prompt(resp: str) -> str:
    resp = (resp or "").strip()
    return (
        f"PII_TYPES = {PII_TYPES}\n"
        f"Below is a numbered list of ONLY JSON snippet(s) extracted from a HAR or HTML capture.\n"
        f"Does any snippet contain any of the user PII from PII_TYPES and if so, which ones.\n"
        f"Give me results as Y,{{label1 (example1),label2 (example2)}} for detected PII and N,{{}} for not detected PII.\n"
        f"For each flagged label, include exactly one example value in parentheses from the JSON snippet.\n"
        f"Make sure to only flag the USER PII and not website or company information:\n\n"
        f"RESPONSES:\n{resp}\n\n"
        f"Answer:"
    )

# ─── Normalization helpers ────────────────────────────────────────────────────
_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*")
_YN_RE    = re.compile(r"^\s*([YyNn])\s*,\s*(.*)$")
def _ascii_quotes(s: str) -> str:
    return (s.replace("\u2018", "'").replace("\u2019", "'")
              .replace("\u201C", '"').replace("\u201D", '"'))

def _simplify_label(s: str) -> str:
    if s is None: return ""
    s = _ascii_quotes(s)
    s = unicodedata.normalize("NFKC", s).strip()
    if s.startswith("{") and s.endswith("}"): s = s[1:-1]
    s = s.replace("{", " ").replace("}", " ")
    if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in ("'", '"')): s = s[1:-1]
    s = _PAREN_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()

ALIASES: Dict[str, str] = {}
for p in PII_TYPES:
    ALIASES[p.lower()] = p
    ALIASES[_simplify_label(p)] = p

def _match_label(token: str) -> Optional[str]:
    if not token: return None
    raw = _ascii_quotes(str(token)).strip().strip(",")
    if not raw: return None
    t_lc = raw.lower()
    if t_lc in CANONICALS: return CANONICALS[t_lc]
    t_simpl = _simplify_label(raw)
    if t_simpl in ALIASES: return ALIASES[t_simpl]
    return None

def _normalize_for_similarity(s: str) -> str:
    s = _simplify_label(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _token_words(s: str) -> set: return set(_normalize_for_similarity(s).split())

def _word_jaccard(a: str, b: str) -> float:
    A, B = _token_words(a), _token_words(b)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

PII_SIMPLIFIED: Dict[str, str] = { _normalize_for_similarity(p): p for p in PII_TYPES }

def _sim_score(a: str, b: str) -> float:
    a_n, b_n = _normalize_for_similarity(a), _normalize_for_similarity(b)
    r = SequenceMatcher(None, a_n, b_n).ratio()
    j = _word_jaccard(a_n, b_n)
    sub = 0.15 if (a_n in b_n or b_n in a_n) and min(len(a_n), len(b_n)) >= 3 else 0.0
    return 0.6*r + 0.35*j + sub

def _fuzzy_match_label(token: str, threshold: float = 0.72, epsilon: float = 0.02, max_hits: int = 2) -> List[str]:
    if not token or not token.strip(): return []
    scores: List[Tuple[float, str]] = []
    for canon_simpl, canon in PII_SIMPLIFIED.items():
        scores.append((_sim_score(token, canon), canon))
    scores.sort(key=lambda x: x[0], reverse=True)
    out: List[str] = []
    if scores and scores[0][0] >= threshold:
        top = scores[0][0]
        for sc, canon in scores:
            if sc + 1e-9 < top - epsilon: break
            if canon not in out: out.append(canon)
            if len(out) >= max_hits: break
    return out

def normalize_and_validate(answer_line: str):
    if answer_line is None: return "N,{}", [], "empty"
    original = str(answer_line); stripped = original.strip()
    if stripped == "": return "N,{}", [], "empty"
    m = _YN_RE.match(stripped)
    if not m: return "N,{}", [], "no_match"
    yn, rest = m.group(1).upper(), m.group(2)
    rest_clean = rest.strip()
    if rest_clean.startswith("{") and rest_clean.endswith("}"):
        rest_clean = rest_clean[1:-1]
    rest_tokens_src = re.sub(r"\s+", " ", rest_clean.replace("{", " ").replace("}", " ")).strip()
    if yn == "N": return "N,{}", [], "parsed"
    cats: List[str] = []; seen = set()
    if rest_tokens_src:
        parts = re.split(r"[;,]", rest_tokens_src)
        for raw_tok in (t for t in parts if t.strip()):
            canonical = _match_label(raw_tok)
            if canonical:
                if canonical not in seen:
                    seen.add(canonical); cats.append(canonical)
                continue
            for cand in _fuzzy_match_label(raw_tok):
                if cand not in seen:
                    seen.add(cand); cats.append(cand)
    if not cats and rest_tokens_src:
        for cand in _fuzzy_match_label(rest_tokens_src, threshold=0.76, epsilon=0.02, max_hits=2):
            if cand not in seen:
                seen.add(cand); cats.append(cand)
    if not cats: return "N,{}", [], "parsed"
    return f"Y,{{{','.join(cats)}}}", cats, "parsed"

# ─── I/O and Pre-Processing (ONLY new [origin] format) ────────────────────────
class _LazyResponses:
    """
    Behaves like a list for tqdm (len), but yields (log_name, preprocessed_text).
    log_name is the original logs filename.
    """
    def __init__(self, paths: List[Path]):
        self.paths = paths
    def __len__(self) -> int:
        return len(self.paths)
    def __iter__(self) -> Iterator[Tuple[str, str]]:
        for p in self.paths:
            try:
                raw = p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"[warn] Could not read {p.name}: {e}", file=sys.stderr)
                continue
            prepped = _only_numbered_json_payloads(raw)
            if not prepped.strip():
                continue
            yield (p.name, prepped)

def _only_numbered_json_payloads(text: str) -> str:
    if not text:
        return ""
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return ""
    out_lines: List[str] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if (i + 1) < len(matches) else len(text)
        payload = (text[start:end] or "").strip()
        if not payload:
            continue
        out_lines.append(payload)
    return "\n".join(out_lines)


def load_log_files() -> List[Tuple[str, str]]:
    if not LOGS_PATH.is_dir():
        print(f"[warn] Logs directory {LOGS_PATH} does not exist or is not a directory.")
        return []
    paths = sorted(LOGS_PATH.glob("*.txt"))
    return _LazyResponses(paths)

# ─── LM Studio Chat Call ──────────────────────────────────────────────────────
def lmstudio_complete(system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
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
        meta: Dict[str, Any] = {"finish_reason": getattr(choice, "finish_reason", None)}
        return text, meta
    except Exception as e:
        return "", {"api_error": f"{type(e).__name__}: {e}"}

# ─── Chunking helpers ─────────────────────────────────────────────────────────
def _split_by_numbered_sections(text: str) -> List[str]:
    starts = [m.start() for m in re.finditer(r"(?m)^\s*\d+\s*:\s*", text)]
    if not starts:
        return [text] if text.strip() else []
    starts.append(len(text))
    chunks = []
    for i in range(len(starts)-1):
        chunks.append(text[starts[i]:starts[i+1]].strip())
    return [c for c in chunks if c]

def _split_long_section(section: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split a single numbered section (e.g., '123: {...}') into multiple chunks
    that each fit within chunk_size. The '<num>: ' header is preserved on every
    chunk, and we apply character overlap on the body to reduce boundary issues.
    """
    if len(section) <= chunk_size:
        return [section]

    # Capture '<num>: ' prefix if present
    m = re.match(r"^\s*(\d+\s*:\s*)", section)
    prefix = m.group(1) if m else ""
    body = section[len(prefix):] if m else section

    body_limit = max(1, chunk_size - len(prefix))
    step = max(1, body_limit - max(0, overlap))

    chunks: List[str] = []
    i = 0
    n = len(body)
    while i < n:
        j = min(n, i + body_limit)
        piece = prefix + body[i:j]
        chunks.append(piece)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks


def _chunk_text(text: str, chunk_size: int = CHUNK_CHAR_LIMIT, overlap: int = CHUNK_CHAR_OVERLAP) -> List[str]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    n = len(text)
    i = 0
    while i < n:
        j = min(n, i + chunk_size)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def _detect_with_chunking(full_text: str) -> Tuple[str, List[str], Dict[str, Any]]:
    chunks = _chunk_text(full_text)
    all_cats: List[str] = []
    outs: List[Dict[str, Any]] = []

    for idx, ch in enumerate(chunks, 1):
        up = build_user_prompt(ch)
        raw_text, meta = lmstudio_complete(SYSTEM_PROMPT, up)
        result_str, cats, parse_status = normalize_and_validate(raw_text)
        outs.append({
            "chunk_index": idx,
            "chars": len(ch),
            "parse_status": parse_status,
            "raw": ("" if raw_text is None else str(raw_text)).strip(),
            "meta": meta,
        })
        for c in cats:
            if c not in all_cats:
                all_cats.append(c)

    agg_result = f"Y,{{{','.join(all_cats)}}}" if all_cats else "N,{}"
    meta = {"chunked": True, "num_chunks": len(chunks), "chunk_outputs": outs}
    return agg_result, all_cats, meta

# ─── Helpers: read existing outputs and decide header ─────────────────────────
def _read_existing_processed(jsonl_path: Path, csv_path: Path) -> Set[str]:
    seen: Set[str] = set()
    if jsonl_path.exists():
        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        img = str(obj.get("log_name", "")).strip()
                        if img:
                            seen.add(img)
                    except Exception:
                        continue
        except Exception as e:
            print(f"[warn] Could not read existing JSONL {jsonl_path}: {e}", file=sys.stderr)
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img = str(row.get("log_name", "")).strip()
                    if img:
                        seen.add(img)
        except Exception as e:
            print(f"[warn] Could not read existing CSV {csv_path}: {e}", file=sys.stderr)
    return seen

def _csv_needs_header(csv_path: Path) -> bool:
    try:
        return (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    except Exception:
        return True

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

    data = load_log_files()
    if not data:
        print("[warn] No rows found in logs/ (expecting *.txt from extract_responses.py).")
        return

    already_done = _read_existing_processed(JSONL_OUT_PATH, CSV_OUT_PATH)
    if already_done:
        print(f"[info] Found {len(already_done)} previously processed file(s). Skipping those.")

    csv_write_header = _csv_needs_header(CSV_OUT_PATH)

    with JSONL_OUT_PATH.open("a", encoding="utf-8") as jout, \
         CSV_OUT_PATH.open("a", encoding="utf-8", newline="") as fcsv:

        csv_writer = csv.DictWriter(fcsv, fieldnames=["log_name", "result"])
        if csv_write_header:
            csv_writer.writeheader()

        seen_this_run: Set[str] = set()

        for log_name, text in tqdm(data, desc="Querying LM Studio"):
            if log_name in already_done or log_name in seen_this_run:
                continue
            seen_this_run.add(log_name)

            original_output = ""
            stripped_output = ""
            model_meta: Dict[str, Any] = {}
            parse_status = "parsed"
            cats: List[str] = []
            result_str = "N,{}"

            if len(text) > CHUNK_CHAR_LIMIT:
                result_str, cats, meta = _detect_with_chunking(text)
                model_meta = meta
                stripped_output = "[chunked outputs in model_meta.chunk_outputs]"
                original_output = stripped_output
            else:
                user_prompt_full = build_user_prompt(text)
                raw_text, meta = lmstudio_complete(SYSTEM_PROMPT, user_prompt_full)
                original_output = "" if raw_text is None else str(raw_text)
                stripped_output  = original_output.strip()
                result_str, cats, parse_status = normalize_and_validate(original_output)
                model_meta = meta

                api_err = (meta or {}).get("api_error", "")
                if api_err and ("context" in api_err.lower() or "overflows" in api_err.lower()):
                    result_str, cats, meta2 = _detect_with_chunking(text)
                    model_meta = {"first_attempt_error": api_err, **meta2}
                    stripped_output = "[chunked outputs in model_meta.chunk_outputs]"
                    original_output = stripped_output
                    parse_status = "parsed"

            record = {
                "log_name": log_name,
                "result": result_str,
                "categories": cats,
                "parse_status": parse_status,
                "model_output": stripped_output,
                "original_output": original_output,
                "model_meta": model_meta,
            }
            if isinstance(model_meta, dict) and "api_error" in model_meta:
                record["parse_status"] = "api_error"

            jout.write(json.dumps(record, ensure_ascii=False) + "\n")
            csv_writer.writerow({"log_name": log_name, "result": record["result"]})
            jout.flush(); fcsv.flush()

    print(f"[done] Appended results to {CSV_OUT_PATH} and {JSONL_OUT_PATH}. Skipped already-processed files.")

if __name__ == "__main__":
    main()