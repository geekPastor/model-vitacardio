from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import os

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
    try:
        df = pd.DataFrame([m.dict() for m in patient_data.mesures])
        print("[DEBUG] DataFrame reçu :\n", df)
        result = analyser_patient_avec_scores(df, pipeline_logreg, rf_model)
        return result
    except Exception as e:
        print("[ERROR] Exception dans /analyser_patient/ :", str(e))
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/test/")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port)
