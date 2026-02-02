#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, cv2, inspect, math, re, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Optional

from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─── Paths ───────────────────────────────────────────────────────────────────
INPUT_DIR   = Path("../ss_dedupte/unique_ss")   # folder of PNGs
OUTPUT_CSV  = Path("ocr.csv")                   # append-only

# ─── Heuristics / thresholds ─────────────────────────────────────────────────
MIN_GOOD_LEN       = 24
TEXT_SCORE_THRESH  = 0.90
MIN_BOX_AREA_FACTOR = 3e-5
ABS_MIN_BOX_AREA    = 150.0

# ─── v3 OCR knobs ────────────────────────────────────────────────────────────
TEXT_DET_SIDE     = 4096
TEXT_DET_LIMITTY  = "max"
TEXT_REC_BS       = 12
USE_TEXTLINE_ORI  = True
USE_DOC_ORI_CLS   = False
USE_DOC_UNWARP    = False

# ─── Tiling ──────────────────────────────────────────────────────────────────
TILE_MAX_SIDE = 3200
TILE_OVERLAP  = 160

# ─── Language sweep & fusion knobs ───────────────────────────────────────────
ENABLE_MULTILINGUAL_FUSION = True
FUSION_MIN_SCORE           = 0.80
FUSION_MIN_LEN             = 16

# Fast-path threshold for "ch" model
CH_FAST_SCORE = 0.95

# ─── Runtime tuning (multiprocessing) ────────────────────────────────────────
def _detect_gpu_ids() -> List[int]:
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            n = paddle.device.cuda.device_count()
            return list(range(n))
    except Exception:
        pass
    return []

VISIBLE_GPU_IDS = _detect_gpu_ids()
DEFAULT_GPU_IDS = VISIBLE_GPU_IDS if VISIBLE_GPU_IDS else [0]

def _parse_gpu_ids_env() -> List[int]:
    s = os.environ.get("OCR_GPU_IDS", "")
    if not s.strip():
        return DEFAULT_GPU_IDS
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return out or DEFAULT_GPU_IDS

GPU_IDS = _parse_gpu_ids_env()
NUM_WORKERS = int(os.environ.get("OCR_NUM_WORKERS", str((len(GPU_IDS)*2) or 2)))
CHUNKSIZE   = int(os.environ.get("OCR_CHUNKSIZE", "2"))

# Reduce Paddle verbosity
os.environ.setdefault("FLAGS_minloglevel", "2")

# ─── Require Paddle ≥3 & import v3 pipeline ──────────────────────────────────
def _require_v3():
    import paddle
    major = int(paddle.__version__.split(".")[0])
    if major < 3:
        raise RuntimeError(
            f"PaddlePaddle >= 3.x required; found {paddle.__version__}."
        )
_require_v3()

from paddleocr import PaddleOCR  # noqa

# ─── Process-local globals ───────────────────────────────────────────────────
_PROCESS_GPU_IDX: Optional[int] = None
_OCR_CACHE: Dict[str, PaddleOCR] = {}
_UNSUPPORTED_LANGS: Set[str] = set()

def _auto_device() -> str:
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception:
        pass
    return "cpu"

def _pin_device_to_process():
    global _PROCESS_GPU_IDX
    try:
        import paddle
        if _PROCESS_GPU_IDX is not None and _auto_device() == "gpu":
            paddle.set_device(f"gpu:{_PROCESS_GPU_IDX}")
        else:
            paddle.set_device("cpu")
    except Exception:
        pass

# ─── Languages (PP-OCRv5 set) ───────────────────────────────────────────────
SUPPORTED_LANGS = [
    "ch","hu","en","rs_latin","fr","id","de","oc","japan","is","korean","lt",
    "chinese_cht","mi","af","ms","it","nl","es","no","bs","pl","pt","sk","cs",
    "sl","cy","sq","da","sv","et","sw","ga","tl","hr","tr","uz","la","ru","be",
    "uk","th","el"
]

# ─── OCR factory ─────────────────────────────────────────────────────────────
def _make_ocr_v3(lang: str) -> PaddleOCR:
    _pin_device_to_process()
    sig = set(inspect.signature(PaddleOCR.__init__).parameters.keys())
    kw = {"lang": lang, "ocr_version": "PP-OCRv5"}
    if "use_textline_orientation" in sig: kw["use_textline_orientation"] = USE_TEXTLINE_ORI
    if "use_doc_orientation_classify" in sig: kw["use_doc_orientation_classify"] = USE_DOC_ORI_CLS
    if "use_doc_unwarping" in sig: kw["use_doc_unwarping"] = USE_DOC_UNWARP
    if "text_det_limit_side_len" in sig: kw["text_det_limit_side_len"] = TEXT_DET_SIDE
    if "text_det_limit_type" in sig: kw["text_det_limit_type"] = TEXT_DET_LIMITTY
    if "text_recognition_batch_size" in sig: kw["text_recognition_batch_size"] = TEXT_REC_BS
    if "device" in sig: kw["device"] = "gpu" if _auto_device() == "gpu" else "cpu"
    try:
        return PaddleOCR(**kw)
    except Exception:
        kw["device"] = "cpu"
        return PaddleOCR(**kw)

def get_ocr(lang: str) -> PaddleOCR:
    if lang in _UNSUPPORTED_LANGS:
        raise RuntimeError(f"Unsupported lang: {lang}")
    if lang not in _OCR_CACHE:
        try:
            _OCR_CACHE[lang] = _make_ocr_v3(lang)
        except Exception:
            _UNSUPPORTED_LANGS.add(lang)
            raise
    return _OCR_CACHE[lang]

# ─── Geometry helpers ───────────────────────────────────────────────────────
def _poly_area_from_box(box):
    box = np.asarray(box)
    if box.ndim == 1 and box.size == 8:
        box = box.reshape(4, 2)
    if box.ndim == 2 and box.shape[-1] == 2:
        return float(cv2.contourArea(box.astype(np.float32)))
    return 0.0

# ─── Predict & collect ──────────────────────────────────────────────────────
def _collect_from_predict(pred: Any, img_shape: Tuple[int, int]) -> Tuple[str, float]:
    if not isinstance(pred, list) or not pred:
        return "", 0.0
    img_h, img_w = img_shape[:2]
    min_box_area = max(ABS_MIN_BOX_AREA, (img_h * img_w) * MIN_BOX_AREA_FACTOR)
    texts, total, length = [], 0.0, 0
    for item in pred:
        res = getattr(item, "res", None)
        if res is None and isinstance(item, dict):
            res = item
        if not isinstance(res, dict):
            continue
        rec_texts  = (res.get("rec_texts")  or [])
        rec_scores = (res.get("rec_scores") or [None] * len(rec_texts))
        polys = res.get("rec_polys")
        if polys is None:
            boxes = res.get("rec_boxes") or res.get("dt_polys")
            if boxes is not None:
                polys = [np.asarray(b).reshape(-1, 2) for b in boxes]
        if polys is None:
            polys = [None] * len(rec_texts)
        for poly, t, sc in zip(polys, rec_texts, rec_scores):
            if not t: continue
            sc_val = float(sc) if sc is not None else 0.0
            if sc_val < TEXT_SCORE_THRESH: continue
            if poly is not None:
                try:
                    area = float(cv2.contourArea(np.asarray(poly, dtype=np.float32)))
                except Exception:
                    area = _poly_area_from_box(poly)
                if area < min_box_area: continue
            l = len(t)
            texts.append(t); length += l; total += sc_val * l
    if not texts:
        return "", 0.0
    return " ".join(texts).strip(), (total / max(1, length))

def _predict_with_fallback(ocr: PaddleOCR, img_rgb: np.ndarray) -> Tuple[str, float]:
    try:
        out = ocr.predict(img_rgb)
        t, s = _collect_from_predict(out, img_rgb.shape[:2])
        if t: return t, s
    except Exception: pass
    try:
        legacy = ocr.ocr(img_rgb, cls=False)
        img_h, img_w = img_rgb.shape[:2]
        min_box_area = max(ABS_MIN_BOX_AREA, (img_h * img_w) * MIN_BOX_AREA_FACTOR)
        texts, total, length = [], 0.0, 0
        for page in legacy or []:
            for det in page or []:
                if not isinstance(det, (list, tuple)) or len(det) < 2: continue
                geom, maybe = det[0], det[1]
                t, sc = (maybe[0], maybe[1]) if isinstance(maybe, (list, tuple)) and len(maybe)>=2 else (None,None)
                if not t: continue
                sc_val = float(sc) if sc is not None else 0.0
                if sc_val < TEXT_SCORE_THRESH: continue
                if geom is not None:
                    area = _poly_area_from_box(geom)
                    if area < min_box_area: continue
                l = len(t); texts.append(t); length += l; total += sc_val * l
        if texts:
            return " ".join(texts).strip(), (total / max(1, length))
    except Exception: pass
    return "", 0.0

# ─── Image IO & tiling ───────────────────────────────────────────────────────
def _np_from_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))

def _tiles_for_size(w: int, h: int, max_side: int, overlap: int):
    xs, x = [], 0
    while True:
        xe = min(x+max_side, w); xs.append((x,xe))
        if xe == w: break
        x = xe - overlap
    ys, y = [], 0
    while True:
        ye = min(y+max_side, h); ys.append((y,ye))
        if ye == h: break
        y = ye - overlap
    return [(x0,y0,x1,y1) for (x0,x1) in xs for (y0,y1) in ys]

def _needs_tiling(p: Path) -> bool:
    with Image.open(p) as im:
        w,h = im.size
    return (w > TILE_MAX_SIDE) or (h > TILE_MAX_SIDE)

# ─── Scoring & fusion ───────────────────────────────────────────────────────
_SCRIPT_BLOCKS = {
    "cjk": re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]"),
    "latin": re.compile(r"[A-Za-z]"),
    "arabic": re.compile(r"[\u0600-\u06FF]"),
    "cyril": re.compile(r"[\u0400-\u04FF]"),
    "thai": re.compile(r"[\u0E00-\u0E7F]"),
    "greek": re.compile(r"[\u0370-\u03FF]"),
}

def _script_sig(text: str) -> Set[str]:
    sig=set()
    for k,rx in _SCRIPT_BLOCKS.items():
        if rx.search(text): sig.add(k)
    return sig

def _utility_score(text: str, avg_score: float) -> float:
    eff_len = max(0,len(text))
    return avg_score*math.log1p(eff_len)

def _dedup_words(text: str) -> str:
    toks = re.split(r"\s+", text.strip())
    seen,out=set(),[]
    for t in toks:
        k=t.lower()
        if k not in seen:
            out.append(t); seen.add(k)
    return " ".join(out)

# ─── OCR per image ───────────────────────────────────────────────────────────
def _ocr_once(img_path: Path, lang: str) -> Tuple[str,float]:
    try:
        ocr=get_ocr(lang)
    except Exception:
        return "",0.0
    if _needs_tiling(img_path):
        arr=_np_from_image(img_path); h,w=arr.shape[:2]
        tiles=_tiles_for_size(w,h,TILE_MAX_SIDE,TILE_OVERLAP)
        texts,total,length=[],0.0,0
        for (x0,y0,x1,y1) in tiles:
            tile=arr[y0:y1,x0:x1,:]
            t,s=_predict_with_fallback(ocr,tile)
            if t:
                l=len(t); texts.append(t); length+=l; total+=s*l
        if not texts: return "",0.0
        return " ".join(texts).strip(), (total/max(1,length))
    arr=_np_from_image(img_path)
    return _predict_with_fallback(ocr,arr)

def best_ocr_for_image(img: Path) -> Tuple[str,List[str],List[Tuple[str,float,str]]]:
    audit=[]
    # 1) Fast path: "ch"
    t_ch,s_ch=_ocr_once(img,"ch")
    audit.append(("ch",s_ch,(t_ch[:160]+"...") if len(t_ch)>160 else t_ch))
    if s_ch>=CH_FAST_SCORE:
        return t_ch,["ch"],audit
    # 2) Full sweep (excluding ch since already tried)
    tried=[("ch",t_ch,s_ch)]
    for lang in SUPPORTED_LANGS:
        if lang=="ch": continue
        t,s=_ocr_once(img,lang)
        tried.append((lang,t,s))
        audit.append((lang,s,(t[:160]+"...") if len(t)>160 else t))
    ranked=sorted(tried,key=lambda x:_utility_score(x[1],x[2]),reverse=True)
    if not ranked or (not ranked[0][1] and ranked[0][2]<=0):
        return "",[],audit
    best_lang,best_text,best_score=ranked[0]
    final_text=best_text; contrib=[best_lang]
    if ENABLE_MULTILINGUAL_FUSION and len(ranked)>=2:
        lang2,text2,score2=ranked[1]
        if text2:
            sig1=_script_sig(best_text); sig2=_script_sig(text2)
            if sig1.isdisjoint(sig2) and best_score>=FUSION_MIN_SCORE and score2>=FUSION_MIN_SCORE \
               and len(best_text)>=FUSION_MIN_LEN and len(text2)>=FUSION_MIN_LEN:
                final_text=_dedup_words(best_text+" "+text2)
                contrib=[best_lang,lang2]
    return final_text,contrib,audit[:8]

# ─── Worker init & per-image task ───────────────────────────────────────────
def _worker_init(gpu_id:int):
    global _PROCESS_GPU_IDX,_OCR_CACHE,_UNSUPPORTED_LANGS
    _PROCESS_GPU_IDX=gpu_id; _OCR_CACHE={}; _UNSUPPORTED_LANGS=set()
    _pin_device_to_process()

def _process_image(img_path_str:str)->Tuple[str,str]:
    img_path=Path(img_path_str)
    try:
        final_text,_,_ = best_ocr_for_image(img_path)
        return img_path.name,(final_text or "")
    except Exception as e:
        return img_path.name,f"ERROR: {e}"

# ─── CSV I/O ─────────────────────────────────────────────────────────────────
def read_done(csv_path: Path) -> Set[str]:
    out=set()
    if not csv_path.is_file(): return out
    with csv_path.open("r",newline="",encoding="utf-8") as f:
        r=csv.DictReader(f)
        if r.fieldnames and "image_name" in r.fieldnames:
            for row in r:
                n=(row.get("image_name") or "").strip()
                if n: out.add(n)
    return out

def open_csv(csv_path: Path):
    new=not csv_path.exists()
    f=csv_path.open("a",newline="",encoding="utf-8")
    w=csv.writer(f)
    if new:
        w.writerow(["image_name","extracted_text"])
    return f,w

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    try: mp.set_start_method("spawn",force=True)
    except RuntimeError: pass
    pngs=sorted(INPUT_DIR.glob("*.png"))
    processed=read_done(OUTPUT_CSV)
    imgs=[str(p) for p in pngs if p.name not in processed]
    if not imgs:
        print(f"No new images in {INPUT_DIR.resolve()}")
        return
    if len(GPU_IDS)==0:
        gpu_assignments=[None]*max(1,NUM_WORKERS)
    else:
        gpu_assignments=[GPU_IDS[i%len(GPU_IDS)] for i in range(max(1,NUM_WORKERS))]
    fcsv,writer=open_csv(OUTPUT_CSV)
    with fcsv:
        with ProcessPoolExecutor(max_workers=max(1,NUM_WORKERS),
                                 initializer=_worker_init,
                                 initargs=(gpu_assignments[0],)) as executor:
            warmups=[executor.submit(_worker_init,g) for g in gpu_assignments]
            for w in warmups:
                try: w.result()
                except Exception: pass
            futures={executor.submit(_process_image,img):img for img in imgs}
            for fut in tqdm(as_completed(futures),total=len(futures),desc="images",unit="img"):
                name,text=fut.result()
                writer.writerow([name,text]); fcsv.flush()
    print(f"Saved results to {OUTPUT_CSV.resolve()}")

if __name__=="__main__":
    main()