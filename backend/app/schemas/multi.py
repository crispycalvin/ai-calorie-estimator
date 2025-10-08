from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel, Field

BBox = Tuple[int, int, int, int]  # x, y, w, h in pixels

class ItemEstimate(BaseModel):
    bbox: BBox
    detector_label: Optional[str] = None
    dish: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    calories: Optional[float] = None
    macros: Optional[Dict[str, float]] = None
    warnings: List[str] = Field(default_factory=list)

class MultiEstimateResponse(BaseModel):
    ok: bool = True
    items: List[ItemEstimate] = Field(default_factory=list)
    image_size: Tuple[int, int]
    notes: List[str] = Field(default_factory=list)
