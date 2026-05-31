from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

def train_ml_models(training_csv, symptom_cols, target_col):
    df = pd.read_csv(training_csv)
    X = df[symptom_cols]
    y = np.ravel(df[[target_col]])

    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X, y)

    nb_model = GaussianNB()
    nb_model.fit(X, y)
    
    return rf_model, nb_model

def predict_symptoms(models, symptoms, symptom_list):
    input_vec = [1 if s in symptoms else 0 for s in symptom_list]
    rf_pred = models[0].predict([input_vec])[0]
    nb_pred = models[1].predict([input_vec])[0]
    return rf_pred, nb_pred
