from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Literal
from model import display_metrics, predict_survival
import io
import sys

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict Titanic survival using Random Forest, XGBoost, and Gradient Boosting",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "name": "Titanic Survival Prediction API",
        "version": "1.0.0",
        "endpoints": ["/health", "/metrics", "/predict", "/docs"]
    }

class PassengerInput(BaseModel):
    age: int
    sex: Literal["male", "female"]
    pclass: Literal[1, 2, 3]
    sibsp: int
    parch: int
    fare: float
    embarked: Literal["S", "C", "Q"]
    location: str
    remote_work: Literal["Yes", "No"]
    certifications: int
    job_title: str

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "age": 30,
                "sex": "male",
                "pclass": 1,
                "sibsp": 0,
                "parch": 0,
                "fare": 50.0,
                "embarked": "S",
                "location": "Australia",
                "remote_work": "No",
                "certifications": 2,
                "job_title": "Data Analyst"
            }]
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics():
    buffer = io.StringIO()
    sys.stdout = buffer
    display_metrics()
    sys.stdout = sys.__stdout__
    return {"metrics": buffer.getvalue()}


@app.post("/predict")
def predict(
    passenger: PassengerInput,
):
    result = predict_survival(passenger.model_dump())
    return {
        "input": passenger.model_dump(),
        "predictions": result
    }