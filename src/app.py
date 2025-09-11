from fastapi import FastAPI
from pydantic import BaseModel
from src.hybrid_infer import hybrid_predict  # adjust import if necessary

app = FastAPI()

class PostIn(BaseModel):
    content_text: str

class PredictionOut(BaseModel):
    predicted_label: int
    confidence: float
    flagged: bool
    language: str
    model_used: str

@app.post("/predict", response_model=PredictionOut)
def predict_post(post: PostIn):
    result = hybrid_predict(post.content_text)  # returns dict
    return PredictionOut(**result)
