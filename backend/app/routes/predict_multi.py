from fastapi import APIRouter, UploadFile, File, HTTPException
from ..ml.pipeline_multi import run_inference_multi
from ..schemas.multi import MultiEstimateResponse

router = APIRouter()

@router.post("/estimate/multi", response_model=MultiEstimateResponse)
async def estimate_multi(file: UploadFile = File(...)) -> MultiEstimateResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (content-type image/*).")
    content = await file.read()
    return MultiEstimateResponse(**run_inference_multi(content))
