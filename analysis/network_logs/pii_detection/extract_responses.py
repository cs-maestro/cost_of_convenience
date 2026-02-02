#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, base64, json, re, hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable
from tqdm import tqdm

# ========= Config =========
HAR_HTML_DIR = Path("./har_html_data")   # contains <hash>.har, <hash>.server.html, <hash>.dom.html
LOGS_DIR     = Path("./logs")            # final per-hash output: logs/<hash>.txt

# ========= Helpers =========
def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()

def _headers_map(headers: Any) -> Dict[str, str]:
    m: Dict[str, str] = {}
    if isinstance(headers, list):
        for h in headers:
            name = (h.get("name") or "").lower()
            val  = (h.get("value") or "")
            if name:
                m[name] = val
    elif isinstance(headers, dict):
        for k, v in headers.items():
            m[str(k).lower()] = str(v)
    return m

def _content_type(resp: Dict[str, Any]) -> str:
    ctype = (resp.get("content", {}) or {}).get("mimeType") or ""
    if ctype:
        return ctype.lower()
    h = _headers_map(resp.get("headers", []))
    return h.get("content-type", "").lower()

def _charset_from_content_type(ct: str) -> Optional[str]:
    if not ct:
        return None
    parts = [p.strip() for p in ct.split(";")]
    for p in parts[1:]:
        if p.lower().startswith("charset="):
            return p.split("=", 1)[1].strip().strip('"').strip("'")
    return None

def _decode_body(text: Any, encoding: Optional[str], charset: Optional[str]) -> str:
    if text is None:
        return ""
    if (encoding or "").lower() == "base64":
        try:
            raw = base64.b64decode(text, validate=False)
        except Exception:
            return str(text)
        for cs in [charset, "utf-8", "utf-16", "latin-1"]:
            if not cs:
                continue
            try:
                return raw.decode(cs, errors="replace")
            except Exception:
                continue
        return raw.decode("utf-8", errors="replace")
    return str(text)

_JSONP_RE = re.compile(
    r"""^\s*([a-zA-Z_$][\w$]*)\s*\(\s*(?P<payload>\{[\s\S]*\}|\[[\s\S]*\])\s*\)\s*;?\s*$""",
    re.MULTILINE,
)
def _unwrap_jsonp(body: str) -> Optional[str]:
    m = _JSONP_RE.match(body)
    return m.group("payload") if m else None

def _looks_like_json(ct: str, body: str) -> bool:
    ct_l = (ct or "").lower()
    if "json" in ct_l:
        return True
    s = body.lstrip()
    return s.startswith("{") or s.startswith("[")

def _canonical_minified_json(parsed: Any) -> str:
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

def _looks_like_html(ct: str, body: str) -> bool:
    ct_l = (ct or "").lower()
    if "text/html" in ct_l:
        return True
    s = body.lstrip().lower()
    return s.startswith("<!doctype html") or s.startswith("<html") or "<html" in s[:2000]

# ========= JSON extractors for HTML =========
_SCRIPT_TAG_RE = re.compile(
    r"""<script\b(?P<attrs>[^>]*)>(?P<code>[\s\S]*?)</script>""",
    re.IGNORECASE
)
_SCRIPT_TYPE_JSON_RE = re.compile(
    r"""type\s*=\s*["']\s*(?:application/(?:ld\+json|json|manifest\+json))\s*["']""",
    re.IGNORECASE
)
_SCRIPT_ID_NEXT_RE = re.compile(
    r'''\bid\s*=\s*["']__NEXT_DATA__["']''',
    re.IGNORECASE
)
_INLINE_ASSIGN_JSON_RE = re.compile(
    r"""(?:
            (?:window|self|globalThis|\w+)\.[A-Za-z0-9_.$]+\s*=\s*
          | (?:var|let|const)\s+[A-Za-z0-9_$]+\s*=\s*
        )
        (?P<json>\{[\s\S]*?\}|\[[\s\S]*?\])\s*;""",
    re.IGNORECASE | re.VERBOSE
)
_CDATA_WRAP_RE = re.compile(r"""^\s*//\s*<!\[CDATA\[\s*|\s*//\s*\]\]>\s*$""", re.MULTILINE)

def _extract_json_from_html(html: str) -> Iterable[str]:
    if not html:
        return
    # 1) JSON script tags (by type) OR explicit __NEXT_DATA__ id
    for m in _SCRIPT_TAG_RE.finditer(html):
        attrs = m.group("attrs") or ""
        code = m.group("code") or ""
        if not (_SCRIPT_TYPE_JSON_RE.search(attrs) or _SCRIPT_ID_NEXT_RE.search(attrs)):
            continue
        code = _CDATA_WRAP_RE.sub("", code).strip()
        if not code:
            continue
        try:
            parsed = json.loads(code)
            yield _canonical_minified_json(parsed)
        except Exception:
            continue
    # 2) Inline assignments with strict-JSON RHS
    for m in _INLINE_ASSIGN_JSON_RE.finditer(html):
        chunk = (m.group("json") or "").strip()
        if not chunk:
            continue
        try:
            parsed = json.loads(chunk)
            yield _canonical_minified_json(parsed)
        except Exception:
            continue

# ========= Main =========
def main():
    if not HAR_HTML_DIR.is_dir():
        print(f"[fatal] Input directory not found: {HAR_HTML_DIR}", file=sys.stderr)
        sys.exit(2)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Build the set of hashes (stems) from whatever is present.
    stems = set(p.stem for p in HAR_HTML_DIR.glob("*.har"))
    stems |= set(p.stem.replace(".server", "") for p in HAR_HTML_DIR.glob("*.server.html"))
    stems |= set(p.stem.replace(".dom", "") for p in HAR_HTML_DIR.glob("*.dom.html"))
    stems = sorted(stems)
    if not stems:
        print("[warn] No HAR/HTML files found in har_html_data/")
        return

    for h in tqdm(stems, desc="Collecting JSON from HAR + HTML"):
        har_path   = HAR_HTML_DIR / f"{h}.har"
        srv_path   = HAR_HTML_DIR / f"{h}.server.html"
        dom_path   = HAR_HTML_DIR / f"{h}.dom.html"

        seen: set[str] = set()
        items: List[tuple[str, str]] = []  # (origin, json)

        # ---- From HAR ----
        if har_path.exists():
            try:
                har_data = json.loads(har_path.read_text(encoding="utf-8"))
                entries = (har_data.get("log", {}) or {}).get("entries", []) or []
                for entry in entries:
                    resp = (entry or {}).get("response", {}) or {}
                    content = resp.get("content", {}) or {}
                    raw_text = content.get("text", "")
                    if raw_text in (None, ""):
                        continue

                    ct = _content_type(resp)
                    charset = _charset_from_content_type(ct)
                    body = _decode_body(raw_text, content.get("encoding"), charset)
                    if not body.strip():
                        continue

                    handled = False
                    if _looks_like_json(ct, body):
                        try:
                            parsed = json.loads(body)
                            canon = _canonical_minified_json(parsed)
                            jh = _hash_text(canon)
                            if jh not in seen:
                                seen.add(jh); items.append(("har:json", canon)); handled = True
                        except Exception:
                            unwrapped = _unwrap_jsonp(body)
                            if unwrapped is not None:
                                try:
                                    parsed = json.loads(unwrapped)
                                    canon = _canonical_minified_json(parsed)
                                    jh = _hash_text(canon)
                                    if jh not in seen:
                                        seen.add(jh); items.append(("har:jsonp", canon)); handled = True
                                except Exception:
                                    pass

                    if not handled and _looks_like_html(ct, body):
                        for canon in _extract_json_from_html(body):
                            jh = _hash_text(canon)
                            if jh not in seen:
                                seen.add(jh); items.append(("har:html", canon))
            except Exception as e:
                print(f"[warn] Could not parse HAR {har_path.name}: {e}", file=sys.stderr)

        # ---- From saved HTMLs ----
        try:
            if srv_path.exists():
                html = srv_path.read_text(encoding="utf-8", errors="replace")
                for canon in _extract_json_from_html(html):
                    jh = _hash_text(canon)
                    if jh not in seen:
                        seen.add(jh); items.append(("html:server", canon))
        except Exception as e:
            print(f"[warn] Could not read {srv_path.name}: {e}", file=sys.stderr)

        try:
            if dom_path.exists():
                html = dom_path.read_text(encoding="utf-8", errors="replace")
                for canon in _extract_json_from_html(html):
                    jh = _hash_text(canon)
                    if jh not in seen:
                        seen.add(jh); items.append(("html:dom", canon))
        except Exception as e:
            print(f"[warn] Could not read {dom_path.name}: {e}", file=sys.stderr)

        # ---- Write only if something was found ----
        if items:
            out_path = LOGS_DIR / f"{h}.txt"
            numbered = "\n".join(
                f"{i} [{origin}] {payload}"
                for i, (origin, payload) in enumerate(items, start=1)
            )
            out_path.write_text(numbered, encoding="utf-8", errors="surrogatepass")
        # else: skip writing any file

    print(f"[ok] Wrote JSON logs (only when non-empty) to {LOGS_DIR.resolve()}")

if __name__ == "__main__":
    main()
