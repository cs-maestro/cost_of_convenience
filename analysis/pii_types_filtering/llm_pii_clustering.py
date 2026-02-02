#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys, unicodedata
from pathlib import Path
from typing import List, Tuple, Dict

LM_STUDIO_URL = "http://127.0.0.1:1234"

# ─── Config (env only; no CLI args) ───────────────────────────────────────────
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", f"{LM_STUDIO_URL}/v1")
LMSTUDIO_API_KEY  = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
LMSTUDIO_MODEL    = os.environ.get("LMSTUDIO_MODEL", "openai/gpt-oss-120b")

LEXICON_PATH      = Path(os.environ.get("PII_LEXICON", "pii_lexicon.txt"))
OUT_PATH          = Path(os.environ.get("PII_OUT", "pii_clusters.txt"))

MAX_TOKENS        = int(float(os.environ.get("MAX_TOKENS", "4096")))
TEMPERATURE       = float(os.environ.get("TEMPERATURE", "0.0"))   # deterministic
TOP_P             = float(os.environ.get("TOP_P", "1.0"))
FREQ_PENALTY      = float(os.environ.get("FREQ_PENALTY", "0.2"))  # discourage repeats
PRES_PENALTY      = float(os.environ.get("PRES_PENALTY", "0.0"))

# If set, we append a final section with any still-missing keywords (keeps original intact)
ALLOW_APPEND      = os.environ.get("APPEND_UNASSIGNED", "0") == "1"

# ─── Prompt (no predefined categories) ────────────────────────────────────────
PROMPT_TEMPLATE = """You are a precise data classification assistant.

Task:
Cluster the PII-related keywords into logical categories of Personally Identifiable Information (PII).

STRICT output format (PRINT ONLY THIS, nothing else):
For each category you create, output exactly:
## <Category Name>
- <keyword 1>
- <keyword 2>
...

Hard rules:
1) Create your own concise category names (1–3 words, letters/spaces/&/hyphens only). No colons or extra text.
2) Use as many categories as needed, but keep them meaningful. Avoid a generic “Miscellaneous” unless truly necessary.
3) Every input keyword MUST appear exactly once across all bullets (no drops, no duplicates, no paraphrasing).
4) Do not sort or alter keywords; copy each keyword string once exactly as provided.
5) Do NOT output any explanations, notes, counts, or code fences. Only the headers and bullet lists.

Keywords (N = {N}):
<<KEYWORDS_START>>
{KEYWORDS}
<<KEYWORDS_END>>
"""

# ─── I/O helpers ──────────────────────────────────────────────────────────────
_SPLIT = re.compile(r"[,\\n]+")

def load_keywords(path: Path) -> List[str]:
    s = path.read_text(encoding="utf-8").strip()
    if not s:
        return []
    if "\n" in s and "," not in s:
        items = [line.strip() for line in s.splitlines() if line.strip()]
    else:
        items = [x.strip() for x in _SPLIT.split(s) if x.strip()]
    # de-dup, preserve order
    seen, out = set(), []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def build_prompt(keywords: List[str]) -> str:
    kw_block = "\n".join(keywords)
    return PROMPT_TEMPLATE.format(N=len(keywords), KEYWORDS=kw_block)

# ─── LM Studio (OpenAI-compatible) ────────────────────────────────────────────
def make_client(base_url: str, api_key: str):
    try:
        from openai import OpenAI
    except Exception:
        print("Please install the OpenAI SDK:  pip install openai", file=sys.stderr)
        raise
    return OpenAI(base_url=base_url, api_key=api_key)

def lmstudio_complete(client, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content":
             "You are a precise data classification assistant. "
             "Follow the user's constraints exactly and print only the required structure."},
            {"role": "user",   "content": prompt},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        frequency_penalty=FREQ_PENALTY,
        presence_penalty=PRES_PENALTY,
        stream=False,
    )
    choice = resp.choices[0] if resp.choices else None
    return (choice.message.content or "") if choice and choice.message else ""

# ─── Helpers ─────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u00A0", " ")
    s = _WS.sub(" ", s.strip()).lower()
    return s

HEADER_RE = re.compile(r"^\s*##\s+(.+?)\s*$", re.M)
BULLET_RE = re.compile(r"^\s*-\s+(.*\S)\s*$", re.M)

def extract_bulleted_keywords(output_text: str) -> List[str]:
    return [m.group(1) for m in BULLET_RE.finditer(output_text or "")]

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not LEXICON_PATH.exists():
        print(f"[ERROR] Missing lexicon: {LEXICON_PATH}", file=sys.stderr); sys.exit(1)

    keywords = load_keywords(LEXICON_PATH)
    if not keywords:
        print("[ERROR] No keywords found.", file=sys.stderr); sys.exit(1)

    prompt = build_prompt(keywords)

    client = make_client(LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY)
    try:
        _ = client.models.list()
    except Exception as e:
        print(f"[FATAL] Could not reach LM Studio at {LMSTUDIO_BASE_URL}. Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Querying LM Studio model '{LMSTUDIO_MODEL}' …")
    try:
        text = lmstudio_complete(client, LMSTUDIO_MODEL, prompt)
    except Exception as e:
        print(f"[FATAL] LM Studio API error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)

    # Save EXACT output
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"[OK] Wrote structured clusters to {OUT_PATH}")

if __name__ == "__main__":
    main()
