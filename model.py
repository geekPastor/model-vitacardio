import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, classification_report

# Entraînement du modèle Random Forest
from sklearn.ensemble import RandomForestClassifier

# Chargement des données et entraînement des modèles pour import
_df = pd.read_csv("./dataset.csv")
_X = _df.drop("heartAttack", axis=1)
_y = _df["heartAttack"]
_X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=0.2, random_state=42)
pipeline_logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])
pipeline_logreg.fit(_X_train, _y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(_X_train, _y_train)

del _df, _X, _y, _X_train, _X_test, _y_train, _y_test


if __name__ == "__main__":
    # Chargement des données
    df = pd.read_csv("./dataset.csv")

    # Séparation features / cible
    X = df.drop("heartAttack", axis=1)
    y = df["heartAttack"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline standardisation + modèle
    pipeline_logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ])

    # Entraînement
    pipeline_logreg.fit(X_train, y_train)

    # Prédictions sur test
    y_pred_logreg = pipeline_logreg.predict(X_test)

    # Rapport classification
    print("=== Logistic Regression ===")
    print(classification_report(y_test, y_pred_logreg))

    # Entraînement du modèle Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Prédictions
    y_pred_rf = rf_model.predict(X_test)

    # Rapport classification
    print("=== Random Forest ===")
    print(classification_report(y_test, y_pred_rf))



def analyser_patient_avec_scores(patient_data: pd.DataFrame, logreg_model, rf_model):
    patient_data['timestamp'] = pd.to_datetime(patient_data['timestamp'])
    scores_logreg = logreg_model.predict_proba(patient_data.drop(columns=['timestamp']))[:, 1]
    scores_rf = rf_model.predict_proba(patient_data.drop(columns=['timestamp']))[:, 1]
    preds_logreg = (scores_logreg >= 0.5).astype(int)
    preds_rf = (scores_rf >= 0.5).astype(int)

    anomalies_detectees = []
    for i, row in patient_data.iterrows():
        anomalies = []
        if row["bloodPressure"] > 140:
            anomalies.append("pression_arterielle > 140 (hypertension)")
        if row["heartRate"] > 100:
            anomalies.append("frequence_cardiaque > 100 (tachycardie)")
        if row["o2Saturation"] < 94:
            anomalies.append("saturation_o2 < 94% (hypoxie)")
        if row["bodyTemperature"] > 37.5:
            anomalies.append("temperature_corporelle > 37.5°C (fièvre)")
        if anomalies:
            anomalies_detectees.append({
                "timestamp": row["timestamp"].strftime('%Y-%m-%d %H:%M:%S'),
                "anomalies": anomalies,
                "logRegScore": float(scores_logreg[i]),
                "rfScore": float(scores_rf[i]),
                "logRegPrediction": int(preds_logreg[i]),
                "rfPrediction": int(preds_rf[i])
            })

    summary = {
        "avScoreLogReg": float(scores_logreg.mean()), ## Calcul du score moyen de la regression logistique
        "avScoreRF": float(scores_rf.mean()),
        "measuresRiscLogReg": int(preds_logreg.sum()), ## Calcul du nombre de mesures à risque selon la regression logistique
        "measuresRiscRF": int(preds_rf.sum()), ## Calcul du nombre de mesures à risque selon le modèle Random Forest
        "logRegRiscPercentage": float(100*preds_logreg.mean()),
        "rfRiscPercentage": float(100*preds_rf.mean())
    }

    return {"anomalies": anomalies_detectees, "summary": summary}