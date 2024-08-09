from typing import Any

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from app.services.transformer_classification import get_overall_sentiment

from app.models.predict import PredictRequest, PredictResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification


api_transformers = APIRouter()


class TextRequest(BaseModel):
    text: str


@api_transformers.post("/classify")
async def classify_text(request: TextRequest):
    try:
        lab, score, mean_sentiment, parts, sentiments = get_overall_sentiment(request.text)
        return {"label": lab}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
