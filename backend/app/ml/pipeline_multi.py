from __future__ import annotations
from typing import Dict, Tuple, List
from PIL import Image

from .detect import detect_items
from .pipeline import run_inference as classify_patch, predict_topk  # new import
import math

# Expanded food keywords & synonyms to capture breakfast platter items.
_FOOD_KEYWORDS = [
    # eggs
    "egg", "eggs", "fried egg", "poached egg", "omelet", "omelette",
    # meats
    "bacon", "sausage", "hotdog", "hot dog",
    # carbs / toast
    "toast", "bread", "loaf", "baguette",
    # beans / tomatoes / mushrooms
    "bean", "beans", "baked beans", "tomato", "cherry tomato", "mushroom", "agaric",
    # other common breakfast
    "pancake", "waffle", "hash brown", "hashbrown", "potato",
]

def _crop(img: Image.Image, bbox: Tuple[int, int, int, int]) -> bytes:
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    x, y = max(0, x), max(0, y)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    patch = img.crop((x, y, x2, y2))
    from io import BytesIO
    buf = BytesIO()
    patch.save(buf, format="PNG")
    return buf.getvalue()

def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0: return 0.0
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0

def _merge_overlaps(items: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
    items = sorted(items, key=lambda d: d.get("confidence", 0.0), reverse=True)
    kept: List[Dict] = []
    for it in items:
        bb = tuple(it["bbox"])  # type: ignore
        if any(_iou(bb, tuple(k["bbox"])) > iou_thresh for k in kept):
            continue
        kept.append(it)
    return kept

def _grid_generate(img: Image.Image, tiles: int = 4, overlap: float = 0.4) -> List[Tuple[int,int,int,int]]:
    """Denser overlapping grid to isolate items on a platter."""
    W, H = img.size
    base_w, base_h = W // tiles, H // tiles
    step_w = max(1, int(base_w * (1 - overlap)))
    step_h = max(1, int(base_h * (1 - overlap)))
    boxes = []
    y = 0
    while y < H:
        x = 0
        h = min(base_h, H - y)
        while x < W:
            w = min(base_w, W - x)
            boxes.append((x, y, w, h))
            x += step_w
        y += step_h
    return boxes

def _choose_food_label(topk: List[tuple[str, float]]) -> tuple[str | None, float]:
    """
    From top-k (label, prob), prefer labels containing food keywords.
    Returns (label_or_none, prob).
    """
    # 1) filter to food-like labels
    candidates = [(lab, p) for lab, p in topk if lab and any(k in lab.lower() for k in _FOOD_KEYWORDS)]
    if candidates:
        # pick highest prob among food-like
        return max(candidates, key=lambda t: t[1])
    # 2) fallback to top-1 even if not food (may be 'French loaf' which is still useful)
    return (topk[0][0], topk[0][1]) if topk else (None, 0.0)

def _fallback_grid(img: Image.Image, max_items: int = 6, conf_cut: float = 0.25) -> List[Dict]:
    """Classify overlapping grid crops; pick food-ish, confident ones; merge overlaps."""
    boxes = _grid_generate(img, tiles=4, overlap=0.4)
    candidates: List[Dict] = []
    for bbox in boxes:
        patch = _crop(img, bbox)
        topk = predict_topk(patch, k=5)
        label, conf = _choose_food_label(topk)
        if label and conf >= conf_cut:
            r = classify_patch(patch)   # reuse to get calories/macros/warnings
            r["warnings"].append("Grid fallback with food-aware re-ranking.")
            candidates.append({
                "bbox": bbox,
                "detector_label": "grid",
                "dish": label,
                "confidence": float(conf),
                "calories": r.get("calories"),
                "macros": r.get("macros"),
                "warnings": r.get("warnings", []),
            })
    candidates = _merge_overlaps(candidates, iou_thresh=0.5)
    return candidates[:max_items]

def run_inference_multi(image_bytes: bytes, max_items: int = 6) -> Dict:
    # 1) Try detector first (might still be empty for many breakfasts)
    dets, (W, H), img = detect_items(image_bytes)
    dets = sorted(dets, key=lambda t: t[2], reverse=True)[:max_items]

    items: List[Dict] = []
    for bbox, det_label, det_conf in dets:
        patch_bytes = _crop(img, bbox)
        cls = classify_patch(patch_bytes)
        cls["warnings"].append(f"Detector: {det_label} (conf ~ {det_conf:.2f})")
        items.append({
            "bbox": bbox,
            "detector_label": det_label,
            "dish": cls.get("dish"),
            "confidence": cls.get("confidence", 0.0),
            "calories": cls.get("calories"),
            "macros": cls.get("macros"),
            "warnings": cls.get("warnings", []),
        })

    # 2) Fallback when detector finds nothing useful
    if not items:
        items = _fallback_grid(img, max_items=max_items, conf_cut=0.25)

    notes = [] if items else ["No items detected. Try a closer photo, different angle, or enable food-specific detection."]
    return {"ok": True, "items": items, "image_size": (W, H), "notes": notes}
