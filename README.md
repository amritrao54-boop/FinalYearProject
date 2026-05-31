# Cattle Disease Prediction System

A comprehensive AI-powered application for diagnosing cattle diseases using both clinical symptoms and photographic image analysis.

## 🚀 Features

- **Multi-Modal Diagnosis**: Combine symptom checklists with deep learning image analysis for higher diagnostic confidence.
- **Deep Learning (CNN)**: Utilizes a fine-tuned **MobileNetV2** model trained on ~4,500 real photographic samples of cattle diseases.
- **Symptom Ensemble**: Uses a majority-vote ensemble of **Random Forest**, **Naive Bayes**, and **Decision Tree** models for symptom-based analysis.
- **Intelligent Fallback**: Features a heuristic color/texture analysis system that provides insights even in the absence of a trained model.
- **High Accuracy**: Achieves **90.8% validation accuracy** on real-world photographic datasets.

## 📂 Project Structure

- `app.py`: Main Flask application server.
- `cnn_model.py`: Model loading and image prediction logic (with startup pre-loading).
- `train_cnn.py`: Training script for the image recognition model.
- `data/`: Contains the symptom dataset (`Training_20symptoms.csv`) and the trained model weights.
- `templates/` & `static/`: Web interface assets (HTML/CSS).

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Cattle-Disease-Prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (Optional)**:
   If the weights are missing or you want to retrain with new data:
   ```bash
   python train_cnn.py
   ```

## 🏃 Running the Application

Start the Flask server:
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## 📊 Supported Diseases

The system is specifically optimized for:
- **Lumpy Skin Disease**
- **Foot and Mouth Disease (FMD)**
- **Mastitis**
- **Healthy Cattle**
- *Fallback support for Blackleg and Foot Rot via heuristic analysis.*

## 🧪 Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-Learn
- **Deep Learning**: TensorFlow, Keras (MobileNetV2)
- **Data Handling**: Pandas, NumPy
- **Image Processing**: PIL (Pillow)

---
*Created with the assistance of Antigravity AI.*
