from fastapi import APIRouter
from app.utils.similarity_calculator import calculate_similarity
from app.models.predict import SimilarityResponse, SimilarityRequest

router = APIRouter()


@router.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity_endpoint(request: SimilarityRequest):
    similarity = calculate_similarity(request.method, request.line1, request.line2)
    return {"method": request.method, "line1": request.line1, "line2": request.line2, "similarity": similarity}
