from flask import Flask, request, render_template, redirect, url_for, flash
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from collections import Counter
import os
from werkzeug.utils import secure_filename
from cnn_model import init_model, predict_disease_from_image

# Pre-load the heavy CNN model at startup to prevent connection resets
init_model()

app = Flask(__name__)
app.secret_key = 'cattlehealth_secret_2025'

# ─── Upload configuration ────────────────────────────────────────────────────
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# ─── Symptom-based ML models ─────────────────────────────────────────────────
SYMPTOMS_LIST = [
    'fever', 'coughing', 'depression', 'loss_of_appetite', 'diarrhoea',
    'salivation', 'lameness', 'swelling', 'weight_loss'
]

df = pd.read_csv('data/Training_20symptoms.csv')
X = df[SYMPTOMS_LIST]
y = np.ravel(df[['prognosis']])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X, y_encoded)

nb_model = GaussianNB()
nb_model.fit(X, y_encoded)

dt_model = DecisionTreeClassifier()
dt_model.fit(X, y_encoded)

# ─── Helpers ─────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    results = {}
    final_prediction = None
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        input_vec = [1 if s in symptoms else 0 for s in SYMPTOMS_LIST]

        rf_pred_enc = rf_model.predict([input_vec])[0]
        rf_pred = label_encoder.inverse_transform([rf_pred_enc])[0]
        nb_pred_enc = nb_model.predict([input_vec])[0]
        nb_pred = label_encoder.inverse_transform([nb_pred_enc])[0]
        dt_pred_enc = dt_model.predict([input_vec])[0]
        dt_pred = label_encoder.inverse_transform([dt_pred_enc])[0]

        results = {
            "Random Forest": rf_pred,
            "Naive Bayes": nb_pred,
            "Decision Tree": dt_pred,
        }

        pred_counts = Counter(results.values())
        most_common = pred_counts.most_common(1)[0]
        if most_common[1] > 1:
            final_prediction = most_common[0]

    return render_template('index.html', results=results,
                           final_prediction=final_prediction,
                           symptoms_list=SYMPTOMS_LIST)


@app.route('/image-detection', methods=['GET'])
def image_detection():
    return render_template('image_detection.html',
                           prediction=None, confidence=None,
                           image_url=None, error=None)


@app.route('/predict-image', methods=['POST'])
def predict_image():
    error = None
    prediction = None
    confidence = None
    image_url = None

    if 'image' not in request.files:
        error = "No file part in the request. Please select an image."
    else:
        file = request.files['image']
        if file.filename == '':
            error = "No file selected. Please choose an image to upload."
        elif not allowed_file(file.filename):
            error = "Invalid file type. Please upload a JPG, PNG, BMP, or WebP image."
        else:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            image_url = '/' + save_path.replace('\\', '/')

            try:
                prediction, confidence = predict_disease_from_image(save_path)
            except Exception as e:
                error = f"Prediction failed: {str(e)}"

    return render_template('image_detection.html',
                           prediction=prediction,
                           confidence=confidence,
                           image_url=image_url,
                           error=error)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

