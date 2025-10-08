#1. Detect dish/food in image
#2. Classify dish/food
#3. Estimate portion size
#4. Map to calories/macros

from __future__ import annotations
from io import BytesIO
from typing import Dict, Optional, Tuple
import json, os

import torch
import torch.nn.functional as F
from PIL import Image

try:
    from torchvision import transforms
    from torchvision.models import resnet18, ResNet18_Weights  #newer API
    _USE_WEIGHTS_ENUM = True
except Exception:
    from torchvision import transforms
    from torchvision.models import resnet18                  #older API
    ResNet18_Weights = None
    _USE_WEIGHTS_ENUM = False

_FOOD101_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "food101_resnet18.pt")
_FOOD101_CLASSES = os.path.join(os.path.dirname(__file__), "weights", "food101_classes.json")

_USE_FOOD101 = os.path.exists(_FOOD101_WEIGHTS) and os.path.exists(_FOOD101_CLASSES)
_food101_classes = None

_DEVICE = torch.device("cpu")
_model = None
_categories = None  # ImageNet labels

def _load_model():
    global _model, _categories, _food101_classes
    if _model is not None:
        return _model, _categories

    if _USE_FOOD101:
        # Load ResNet18 and replace head for 101 classes
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT
            base = resnet18(weights=weights)
        except Exception:
            from torchvision.models import resnet18
            base = resnet18(pretrained=True)

        in_feats = base.fc.in_features
        import torch.nn as nn
        base.fc = nn.Linear(in_feats, 101)  # Food-101
        state = torch.load(_FOOD101_WEIGHTS, map_location=_DEVICE)
        base.load_state_dict(state["model"], strict=True)
        base.eval().to(_DEVICE)

        # load classes and attach preprocess
        with open(_FOOD101_CLASSES, "r", encoding="utf-8") as f:
            _food101_classes = json.load(f)
        # use ImageNet normalization
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        base._preprocess = preprocess
        _model = base
        _categories = _food101_classes
        return _model, _categories
    # ---- fallback to ImageNet path (your existing code below) ----
    ...


    if _USE_WEIGHTS_ENUM and ResNet18_Weights is not None:
        weights = ResNet18_Weights.DEFAULT
        _model = resnet18(weights=weights)
        _categories = list(weights.meta.get("categories", []))
        preprocess = weights.transforms()
    else:
        _model = resnet18(pretrained=True)  # older API
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        _categories = None

    _model.eval().to(_DEVICE)
    _model._preprocess = preprocess  # attach for convenience
    return _model, _categories

_KEYWORD_TO_DISH = [
    ("pizza", "pizza"),
    ("cheeseburger", "burger"),
    ("hamburger", "burger"),
    ("hotdog", "hot dog"),
    ("hot dog", "hot dog"),
    ("ice cream", "ice cream"),
    ("icecream", "ice cream"),
    ("spaghetti", "spaghetti"),
    ("carbonara", "spaghetti"),
    ("meat loaf", "meatloaf"),
    ("steak", "steak"),
    ("sushi", "sushi"),
    ("nigiri", "sushi"),
    ("ramen", "ramen"),
    ("noodle", "noodles"),
    ("sandwich", "sandwich"),
    ("bagel", "bagel"),
    ("burrito", "burrito"),
    ("taco", "taco"),
    ("guacamole", "guacamole"),
    ("fried rice", "fried rice"),
    ("omelet", "omelet"),
    ("salad", "salad"),
    ("pancake", "pancakes"),
]

_DISH_TO_CALORIES = {
    "pizza": 285.0,      # 1 slice
    "burger": 354.0,     # 1 burger
    "hot dog": 151.0,    # 1 hot dog
    "ice cream": 273.0,  # 1 cup
    "spaghetti": 221.0,  # 1 cup cooked (plain)
    "meatloaf": 294.0,   # 1 slice
    "steak": 679.0,      # ~12 oz cooked
    "sushi": 200.0,      # 6–8 pieces mixed
    "ramen": 380.0,      # 1 bowl
    "noodles": 220.0,    # 1 cup cooked
    "sandwich": 300.0,   # 1 sandwich
    "bagel": 245.0,      # 1 medium
    "burrito": 560.0,    # 1 burrito
    "taco": 170.0,       # 1 taco
    "guacamole": 230.0,  # 1/2 cup
    "fried rice": 238.0, # 1 cup
    "omelet": 154.0,     # 2 eggs
    "salad": 150.0,      # small, no dressing
    "pancakes": 175.0,   # 2 small
}

def _bytes_to_tensor(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    model, _ = _load_model()
    x = model._preprocess(image).unsqueeze(0)  # (1,3,224,224)
    return x.to(_DEVICE)

def _idx_to_label(idx: int) -> Optional[str]:
    _, categories = _load_model()
    if categories and 0 <= idx < len(categories):
        return categories[idx]
    return None

def _label_to_dish(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    low = label.lower()
    for key, dish in _KEYWORD_TO_DISH:
        if key in low:
            return dish
    return label  # fallback to raw (may be non-food)

def run_inference(image_bytes: bytes) -> Dict:
    model, _ = _load_model()
    x = _bytes_to_tensor(image_bytes)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        confidence = float(conf.item())
        pred_idx = int(idx.item())

    label = _idx_to_label(pred_idx) or f"class_{pred_idx}"
    dish = _label_to_dish(label)

    warnings = []
    calories = None
    if dish in _DISH_TO_CALORIES:
        calories = _DISH_TO_CALORIES[dish]
        warnings.append("Calories assume a typical single serving; portion size not yet estimated.")
    else:
        warnings.append("Predicted label may not be a food or is unknown; calories unavailable.")

    macros = None
    if calories is not None:
        macros = {
            "protein_g": round(calories * 0.15 / 4, 1),
            "carbs_g":   round(calories * 0.55 / 4, 1),
            "fat_g":     round(calories * 0.30 / 9, 1),
        }

    return {
        "dish": dish,
        "calories": calories,
        "confidence": confidence,
        "macros": macros,
        "warnings": warnings,
    }

def predict_topk(image_bytes: bytes, k: int = 5):
    #Return top-k (label, prob) from the ImageNet classifier.
    model, _ = _load_model()
    x = _bytes_to_tensor(image_bytes)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        confs, idxs = torch.topk(probs, k)
    labels = [_idx_to_label(int(i)) or f"class_{int(i)}" for i in idxs]
    return list(zip(labels, [float(c) for c in confs]))
