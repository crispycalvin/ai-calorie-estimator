from fastapi import APIRouter, UploadFile, File, HTTPException
from ..ml.pipeline import run_inference
from ..schemas.prediction import EstimateResponse

router = APIRouter()

@router.post("/estimate", response_model=EstimateResponse)
async def estimate(file: UploadFile = File(...)) -> EstimateResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (content-type image/*).")
    content = await file.read()
    result = run_inference(content)
    return EstimateResponse(ok=True, **result)