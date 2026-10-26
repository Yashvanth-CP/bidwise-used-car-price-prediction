import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

art = joblib.load("model.joblib")
pipe, edges, q_bins, columns = art["pipe"], art["edges"], art["q_bins"], art["columns"]

app = FastAPI(title="BidWise")

class Car(BaseModel):
    brand: str
    model: str
    manufacturing_year: int
    km_driven: float
    fuel_type: str
    transmission_type: str
    bodytype: str
    number_of_owners: int = 1

@app.post("/predict")
def predict(car: Car):
    age = 2026 - car.manufacturing_year
    row = pd.DataFrame([{
        **car.model_dump(), "city": "unknown",
        "car_age": age, "km_per_year": car.km_driven / max(age, 1),
    }])[columns]
    p = pipe.predict(row)[0]
    q = q_bins[np.digitize([p], edges)[0]]
    return {"price": round(float(np.expm1(p))),
            "low": round(float(np.expm1(p - q))),
            "high": round(float(np.expm1(p + q)))}