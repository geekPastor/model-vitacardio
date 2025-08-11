from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd

# Import des modèles et fonction depuis models.py
from model import pipeline_logreg, rf_model, analyser_patient_avec_scores

app = FastAPI()

class Mesure(BaseModel):
    timestamp: str
    bloodPressure: float
    heartRate: float
    o2Saturation: float
    bodyTemperature: float

class PatientData(BaseModel):
    mesures: List[Mesure]

@app.post("/analyser_patient/")
def analyser(patient_data: PatientData):
    df = pd.DataFrame([m.dict() for m in patient_data.mesures])
    result = analyser_patient_avec_scores(df, pipeline_logreg, rf_model)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    


