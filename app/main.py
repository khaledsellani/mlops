import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from sklearn.datasets import load_digits

app = FastAPI()

# load model
model = joblib.load("artifacts/model.pkl")

# labels
digits = load_digits()

class InputData(BaseModel):
    features: conlist(float, min_length=64, max_length=64)

@app.get("/health")
def root():
    return {"message": "Digits MLOps API is running"}

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)[0]

    return {
        "prediction": int(prediction),
        "label": str(prediction)
    }