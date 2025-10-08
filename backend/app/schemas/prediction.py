"""
Pydantic model used to structure/validate response data 
for calorie estimation API
"""
from typing import Optional, Dict, List
from pydantic import BaseModel, Field

class EstimateResponse(BaseModel):
    ok: bool = Field(True, description = "Request succeeded")
    dish: Optional[str] = Field(None, description = "Predicted dish or food label")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description = "Model confidence in [0,1]")
    calories: Optional[float] = Field(None, description = "Estimated calories (kcal) for assumed portion")
    macros: Optional[Dict[str, float]] = Field(default = None, description = "Optional macro grams: protein, carbs, fat")
    warnings: List[str] = Field(default_factory = list, description = "Any caveats or assumptions about the estimate")
