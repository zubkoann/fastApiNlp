from typing import Any

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from app.services.transformer_classification import get_overall_sentiment

from app.models.predict import PredictRequest, PredictResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

api_router = APIRouter()


class TextRequest(BaseModel):
    text: str


@api_router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, payload: PredictRequest) -> Any:
    """
    ML Prediction API
    """
    input_text = payload.input_text
    model = request.app.state.model

    predict_value = model.predict(input_text)
    return PredictResponse(result=predict_value)
