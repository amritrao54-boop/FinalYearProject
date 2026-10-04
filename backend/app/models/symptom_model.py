import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter

SYMPTOMS_LIST = [
    'fever', 'coughing', 'depression', 'loss_of_appetite', 'diarrhoea',
    'salivation', 'lameness', 'swelling', 'weight_loss'
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Training_20symptoms.csv')

_rf_model = None
_nb_model = None
_dt_model = None
_label_encoder = None

def init_symptom_models():
    global _rf_model, _nb_model, _dt_model, _label_encoder
    if _rf_model is not None:
        return

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df[SYMPTOMS_LIST]
    y = np.ravel(df[['prognosis']])

    _label_encoder = LabelEncoder()
    y_encoded = _label_encoder.fit_transform(y)

    _rf_model = RandomForestClassifier(n_estimators=100)
    _rf_model.fit(X, y_encoded)

    _nb_model = GaussianNB()
    _nb_model.fit(X, y_encoded)

    _dt_model = DecisionTreeClassifier()
    _dt_model.fit(X, y_encoded)

def predict_disease_from_symptoms(symptoms: list):
    init_symptom_models()

    input_vec = [1 if s in symptoms else 0 for s in SYMPTOMS_LIST]
    
    rf_pred_enc = _rf_model.predict([input_vec])[0]
    rf_pred = _label_encoder.inverse_transform([rf_pred_enc])[0]
    
    nb_pred_enc = _nb_model.predict([input_vec])[0]
    nb_pred = _label_encoder.inverse_transform([nb_pred_enc])[0]
    
    dt_pred_enc = _dt_model.predict([input_vec])[0]
    dt_pred = _label_encoder.inverse_transform([dt_pred_enc])[0]

    results = {
        "Random Forest": rf_pred,
        "Naive Bayes": nb_pred,
        "Decision Tree": dt_pred,
    }

    pred_counts = Counter(results.values())
    most_common = pred_counts.most_common(1)[0]
    final_prediction = most_common[0]

    return {
        "prediction": final_prediction,
        "probabilities": results,
        "model": "Ensemble (RF, NB, DT)"
    }
