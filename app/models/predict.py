from pydantic import BaseModel, Field, StrictStr


class PredictRequest(BaseModel):
    input_text: StrictStr = Field(..., title="input_text", description="Input text", example="Input text for ML")


class PredictResponse(BaseModel):
    result: float = Field(..., title="result", description="Predict value", example=0.9)


class SimilarityRequest(BaseModel):
    method: str
    line1: str
    line2: str


class SimilarityResponse(BaseModel):
    method: str
    line1: str
    line2: str
    similarity: float


class PreprocessRequest(BaseModel):
    remove_html: bool
    remove_url: bool
    tokenize: bool
    remove_stopwords: bool
    lemmatize: bool
    lowercasing: bool
    remove_punctuation: bool
    remove_extra_ws: bool
    remove_frequent_words: bool
    spelling_correction: bool
