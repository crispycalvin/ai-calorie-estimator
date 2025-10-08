from __future__ import annotations
from io import BytesIO
import numpy as np
from typing import List, Tuple
from PIL import Image
from ultralytics import YOLO  # type: ignore

_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO("yolov8n.pt")  # auto-download, CPU ok
    return _MODEL

def load_image(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")

def detect_items(image_bytes: bytes, conf_threshold: float = 0.10):
    """
    Returns:
      dets: [((x,y,w,h), class_name, conf), ...],
      (W,H): original image size,
      img: PIL.Image
    """
    model = _get_model()
    img = load_image(image_bytes)
    W, H = img.size
    results = model.predict(source=np.array(img), verbose=False, conf=conf_threshold, device="cpu")

    dets = []
    for r in results:
        names = r.names
        for b in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            conf = float(b.conf[0].item())
            cls_idx = int(b.cls[0].item())
            cls_name = names.get(cls_idx, str(cls_idx))
            dets.append(((x1, y1, bw, bh), cls_name, conf))
    return dets, (W, H), img
