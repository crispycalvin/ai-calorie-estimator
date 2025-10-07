#1. Detect dish/food in image
#2. Classify dish/food
#3. Estimate portion size
#4. Map to calories/macros

from typing import Dict

def run_inference(image_bytes: bytes) -> Dict:
    return {
        "dish": None,   #e.g. "Lasagna"
        "calories": None,   #Estimated kcal given portion size
        "confidence": 0.0   #Model Confidence (0...1)
    }