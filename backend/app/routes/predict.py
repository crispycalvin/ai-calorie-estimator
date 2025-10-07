from fastapi import APIRouter, UploadFile, File
from ..ml.pipeline import run_inference

router = APIRouter()

@router.post("/estimate")
async def estimate(file: UploadFile = File(...)):
    content = await file.read()
    result = run_inference(content)
    return {"ok": True, "result": result}