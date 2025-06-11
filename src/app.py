# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from model import load_model

app = FastAPI()
model = load_model()  # loads model.joblib from working dir

class PredictRequest(BaseModel):
    lux: float
    temperature: float
    humidity: float
    hour: int
    soil_roll3: float

class PredictResponse(BaseModel):
    soil_pred: float

@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    X = [[
        req.lux,
        req.temperature,
        req.humidity,
        req.hour,
        req.soil_roll3
    ]]
    soil_pred = model.predict(X)[0]
    return PredictResponse(soil_pred=soil_pred)
