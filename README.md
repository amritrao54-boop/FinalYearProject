# 🐄 Multimodal Agentic AI-Based Cattle Disease Diagnosis & Veterinary Assistance System

An enterprise-grade, multimodal Agentic AI platform combining Traditional Machine Learning, Deep Learning Computer Vision, LangGraph Orchestration, RAG Knowledge Retrieval, Multimodal Fusion, Explainable AI (Grad-CAM), and Conversational State Memory.

---

## 📑 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Existing System](#3-existing-system)
4. [Upgraded System](#4-upgraded-system)
5. [Target Architecture](#5-target-architecture)
6. [Agent Architecture & Workflow](#6-agent-architecture--workflow)
7. [Machine Learning & Deep Learning Models](#7-machine-learning--deep-learning-models)
8. [Agent Tools](#8-agent-tools)
9. [RAG (Retrieval-Augmented Generation) Pipeline](#9-rag-retrieval-augmented-generation-pipeline)
10. [Multimodal Fusion Layer](#10-multimodal-fusion-layer)
11. [Generative AI Layer](#11-generative-ai-layer)
12. [Explainable AI (Grad-CAM)](#12-explainable-ai-grad-cam)
13. [Conversational State & Database Memory](#13-conversational-state--database-memory)
14. [API Documentation](#14-api-documentation)
15. [Database Architecture](#15-database-architecture)
16. [Evaluation Framework](#16-evaluation-framework)
17. [Testing Strategy](#17-testing-strategy)
18. [Docker & Containerization](#18-docker--containerization)
19. [Deployment & CI/CD](#19-deployment--cicd)
20. [Limitations](#20-limitations)
21. [Future Improvements](#21-future-improvements)

---

## 1. Project Overview
This project transforms a traditional machine learning and computer vision cattle disease classifier into an autonomous **Multimodal Agentic AI Diagnostic Orchestrator**. The system does not discard existing predictive models; instead, it wraps them as intelligent, callable **AI Tools** that are dynamically invoked and reasoned over by an AI Agent.

---

## 2. Problem Statement
Livestock farming contributes heavily to global agriculture, yet rural farmers frequently face:
- Delayed veterinary assistance and high misdiagnosis rates.
- Disconnected diagnostic modalities (symptom checklists evaluated separately from visual lesion images).
- Lack of explainability behind black-box AI model outputs.
- Absence of longitudinal case history tracking across distinct clinical visits.

---

## 3. Existing System
The legacy application was a monolithic Flask app with isolated workflows:
- **Image-based CNN**: MobileNetV2 standalone image classification.
- **Symptom-based ML**: Separate majority-vote ensemble of Random Forest, Naive Bayes, and Decision Tree.
- **Limitation**: Predictions were displayed independently without cross-modal reasoning, contextual explanation, or RAG verification.

---

## 4. Upgraded System
The upgraded system introduces an autonomous **AI Cattle Health Diagnostic Agent**:
- **Dynamically determines required tools** based on provided inputs (symptoms only, image only, mixed, or general query).
- **Executes RAG retrieval** over veterinary medical knowledge chunks.
- **Performs Multimodal Evidence Fusion** to resolve discrepancies between visual markers and clinical symptoms.
- **Provides Grad-CAM visualizations** for visual interpretability.
- **Retains case history** in a relational database for longitudinal tracking.

---

## 5. Target Architecture

```text
                         ┌───────────────────────┐
                         │   React UI Dashboard  │
                         └───────────┬───────────┘
                                     │
                              User Request
                        (Image + Symptoms + Tag ID)
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    FastAPI Backend    │
                         │  /api/agent/diagnose  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   LangGraph Agent     │
                         │    (State Machine)    │
                         └───────────┬───────────┘
                                     │
                 ┌───────────────────┼────────────────────┐
                 │                   │                    │
                 ▼                   ▼                    ▼
        ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐
        │   Image Tool   │  │  Symptom Tool  │  │ Veterinary RAG  │
        │ MobileNetV2+XAI│  │ RF/NB/DT Model │  │ Knowledge Tool  │
        └───────┬────────┘  └───────┬────────┘  └────────┬────────┘
                │                   │                    │
                └───────────────────┼────────────────────┘
                                    ▼
                         ┌───────────────────────┐
                         │ Multimodal Fusion /   │
                         │  Decision Layer (LLM) │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Structured Assessment │
                         │ + Evidence + Grad-CAM │
                         │ + Recommendations     │
                         └───────────────────────┘
```

---

## 6. Agent Architecture & Workflow
Built using **LangGraph** with explicit state management (`AgentState`):

```text
[START]
   ↓
[Analyze Image Node] ──(Image provided?)──► [Image Tool / MobileNetV2]
   ↓
[Analyze Symptoms Node] ──(Symptoms provided/extracted?)──► [Symptom Tool / ML Ensemble]
   ↓
[Retrieve Knowledge Node] ──► [Veterinary RAG Search]
   ↓
[Multimodal Fusion & Assessment Node] ──► [LLM / Decision Engine]
   ↓
[Structured JSON Response]
   ↓
 [END]
```

---

## 7. Machine Learning & Deep Learning Models
1. **Symptom Ensemble (`symptom_model.py`)**:
   - **Random Forest Classifier** (`n_estimators=100`)
   - **Gaussian Naive Bayes**
   - **Decision Tree Classifier**
   - Trained on structured 20-symptom clinical datasets.
2. **Deep Learning CNN (`image_model.py`)**:
   - **MobileNetV2** transfer learning trained on 224x224 photographic images.
   - Detects visual pathologies: *Lumpy Skin Disease, Foot and Mouth Disease, Mastitis, Foot Rot, Blackleg*.

---

## 8. Agent Tools
All models and data retrieval mechanisms are exposed as standard LangChain tools:
- `analyze_cattle_image(image_path)`: Returns top visual prediction, confidence, and Grad-CAM heatmap path.
- `analyze_cattle_symptoms(symptoms)`: Returns ensemble prediction and individual model breakdown.
- `search_veterinary_knowledge(query)`: Queries veterinary clinical documents.
- `get_case_history(animal_id)`: Fetches historical records for longitudinal tracking.

---

## 9. RAG (Retrieval-Augmented Generation) Pipeline
- **Knowledge Store**: Curated clinical literature covering symptomology, transmission vectors, vaccination protocols, and biosecurity measures.
- **Retriever**: TF-IDF & Cosine Similarity vector store (`vector_store.py`) enabling high-speed, local similarity search without external API dependencies.
- **Grounding**: The LLM fuses facts strictly from retrieved chunks to avoid hallucinations.

---

## 10. Multimodal Fusion Layer
When both image and symptom data are present, the system does not simply average scores:
- Cross-references visual feature confidence with symptom predictions.
- Identifies whether physical lesions corroborate reported clinical signs (e.g., mouth vesicles + lameness corroborating FMD).
- Generates an explanation detailing which modality provided primary evidence.

---

## 11. Generative AI Layer
Uses **Gemini 2.5 Pro** (`ChatGoogleGenerativeAI`) with Pydantic structured output validation (`AssessmentOutput`):
```json
{
  "assessment": {
    "predicted_condition": "Lumpy Skin Disease",
    "confidence": "91%",
    "certainty_level": "Preliminary"
  },
  "evidence": [
    "CNN model identified visual markers characteristic of Lumpy Skin Disease with 91% confidence.",
    "Reported clinical symptoms: fever, skin nodules."
  ],
  "knowledge_sources": [
    "[Lumpy Skin Disease]: LSD is a viral disease causing high fever and nodular skin lesions..."
  ],
  "recommendations": [
    "Isolate affected animal immediately.",
    "Implement vector control and consult a certified veterinarian."
  ],
  "warning": "This is an AI-assisted preliminary assessment and should not replace veterinary examination."
}
```

---

## 12. Explainable AI (Grad-CAM)
- Implements Gradient-weighted Class Activation Mapping (Grad-CAM) on MobileNetV2 using TensorFlow's `GradientTape`.
- Computes gradients of the predicted class score with respect to the feature maps of the final convolutional layer.
- Generates and superimposes a heatmap on the cattle image, visually highlighting the lesion areas that influenced the prediction.

---

## 13. Conversational State & Database Memory
Powered by **SQLAlchemy** relational models (`backend/app/database/models.py`):
- `Animal`: Tag ID, Name, Breed, Age.
- `Case`: Case ID, Timestamp, Status (Open/Resolved).
- `Symptom`: Associated clinical indicators per visit.
- `Prediction`: Model outputs & multimodal assessments.
- `AgentSession`: Historical conversation messages and tool call records.

---

## 14. API Documentation

### `POST /api/agent/diagnose`
Primary multimodal agent endpoint.

**Request Form Data:**
- `user_query` *(string, optional)*: Free-text farmer query or observation.
- `symptoms` *(array of strings, optional)*: List of clinical symptoms.
- `animal_id` *(string, optional)*: Animal Tag ID.
- `image` *(file, optional)*: Cattle photograph (JPG/PNG).

**Interactive Swagger Docs:** `http://localhost:8000/docs`

---

## 15. Database Architecture

```text
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│     Animal      │       │      Case       │       │    Symptom      │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ id (PK)         │◄─────┐│ id (PK)         │◄─────┐│ id (PK)         │
│ name            │      └┼ animal_id (FK)  │      └┼ case_id (FK)    │
│ breed           │       │ created_at      │       │ symptom_list    │
│ age             │       │ status          │       └─────────────────┘
└─────────────────┘       └────────┬────────┘
                                   │
                                   ├────────────────┬─────────────────┐
                                   ▼                ▼                 ▼
                          ┌─────────────────┐ ┌───────────┐ ┌─────────────────┐
                          │   Prediction    │ │AgentSess. │ │  Grad-CAM Img   │
                          ├─────────────────┤ ├───────────┤ ├─────────────────┤
                          │ id (PK)         │ │ id (PK)   │ │ file_path       │
                          │ case_id (FK)    │ │case_id(FK)│ │ timestamp       │
                          │ fusion_pred     │ │messages   │ └─────────────────┘
                          └─────────────────┘ └───────────┘
```

---

## 16. Evaluation Framework
- **ML Classifiers**: Evaluated across Precision, Recall, F1-Score, and Confusion Matrix.
- **Deep Learning CNN**: Per-class accuracy and ROC-AUC on validation splits.
- **RAG & Agent Retrieval**: Retrieval Precision and grounded faithfulness of generated reports.

---

## 17. Testing Strategy
- **Unit Tests**: Verifying individual tool outputs (`image_tool`, `symptom_tool`, `rag_tool`).
- **Integration Tests**: Validating LangGraph state transitions and FastAPI endpoint responses.
- **Edge Cases Tested**: Image-only inputs, symptom-only inputs, missing API keys (fallback validation), and ambiguous symptom phrases.

---

## 18. Docker & Containerization
Run the full-stack system with Docker Compose:
```bash
docker compose up --build
```
Services defined in `docker-compose.yml`:
- `backend`: FastAPI API server on port 8000.
- `frontend`: React Web UI on port 3000.

---

## 19. Deployment & CI/CD
- **Continuous Integration (`.github/workflows/ci.yml`)**:
  - Automatically triggered on push/PR to `main`.
  - Sets up Python environment, installs dependencies, runs test suites, and builds Docker container images.

---

## 20. Limitations
- **Preliminary Assessment**: Output is assistive and non-prescriptive; cannot replace physical palpation, auscultation, or laboratory blood tests.
- **Image Quality Dependence**: Poor lighting, extreme distance, or motion blur may affect CNN accuracy.

---

## 21. Future Improvements
- Integration with edge IoT devices (smart farm collars, temperature sensors).
- Offline mobile app (React Native) with on-device quantized TFLite models.
- Multilingual voice interface for rural farmers in local dialects.

---

## 🏁 Quick Local Run Guide

```bash
# 1. Start Backend Server
python -m pip install -r requirements.txt
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# 2. Start Frontend UI (in /frontend directory)
cd frontend
npm install
npm start
```
Access UI: **http://localhost:3000** | API Docs: **http://localhost:8000/docs**
