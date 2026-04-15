import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Charger le modèle
with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)


class InputData(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {"message": "MLOps API is running"}


@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}
