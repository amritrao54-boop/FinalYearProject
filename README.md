# Multimodal Agentic AI Cattle Health Diagnostic & Veterinary Assistance System

An advanced, multimodal Agentic AI platform for automated preliminary cattle disease diagnosis, veterinary knowledge retrieval (RAG), multimodal evidence fusion, and Explainable AI (Grad-CAM).

---

## 🌟 1. Project Overview & Architectural Evolution

This repository contains the upgraded **Multimodal Agentic AI System** developed from the ground up to unify:
- **Traditional Machine Learning** (Symptom classification ensemble: Random Forest, Naive Bayes, Decision Tree)
- **Computer Vision & Deep Learning** (Transfer learning with MobileNetV2 CNN)
- **Agentic AI Orchestration** (Stateful graph workflows via LangGraph)
- **Tool Calling Mechanism** (Image tool, Symptom tool, Case History tool, RAG tool)
- **Retrieval-Augmented Generation (RAG)** (Veterinary Vector Knowledge Base)
- **Multimodal Decision Layer & Fusion** (Grounding visual + clinical features)
- **Explainable AI (XAI)** (Grad-CAM attention heatmaps)
- **Conversational State & Database Memory** (SQLAlchemy + SQLite/Postgres schemas)
- **Modern REST API** (FastAPI with Pydantic validation)
- **Interactive UI** (React Dashboard with live execution trace)
- **Containerization & CI/CD** (Docker, Docker Compose, GitHub Actions)

---

## 🏗️ 2. High-Level Target Architecture

```text
                               ┌───────────────────────────┐
                               │   React Clinical Frontend │
                               └─────────────┬─────────────┘
                                             │ HTTP POST (Multipart)
                                             ▼
                               ┌───────────────────────────┐
                               │      FastAPI Backend      │
                               │   /api/agent/diagnose     │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   LangGraph Agent Core    │
                               │      (State Machine)      │
                               └─────────────┬─────────────┘
                                             │
                 ┌───────────────────────────┼───────────────────────────┐
                 │                           │                           │
                 ▼                           ▼                           ▼
      ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
      │   Image Tool (CNN)  │     │  Symptom Tool (ML)  │     │  RAG Knowledge Tool │
      │  MobileNetV2 + XAI  │     │ RF / NB / DT Models │     │  TF-IDF / Vector DB │
      └──────────┬──────────┘     └──────────┬──────────┘     └──────────┬──────────┘
                 │                           │                           │
                 └───────────────────────────┼───────────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │  Multimodal Fusion Layer  │
                               │  Decision Ensembling / LLM│
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Structured Diagnosis     │
                               │  + Evidence + Grad-CAM    │
                               │  + Veterinary Guidelines  │
                               └───────────────────────────┘
```

---

## 📂 3. Project Directory Layout

```text
FinalYearProject/
│
├── backend/
│   └── app/
│       ├── main.py                  # FastAPI server entry point & CORS configuration
│       │
│       ├── agent/                   # LangGraph Agent Core
│       │   ├── graph.py             # Compiled StateGraph workflow definition
│       │   ├── state.py             # AgentState TypedDict schema
│       │   └── nodes.py             # Image, Symptom, RAG, and Fusion decision nodes
│       │
│       ├── tools/                   # LangChain Tool Interfaces
│       │   ├── image_tool.py        # Image prediction wrapper tool
│       │   ├── symptom_tool.py      # Symptom classification wrapper tool
│       │   ├── rag_tool.py          # Veterinary knowledge retrieval tool
│       │   └── history_tool.py      # Case history retrieval tool
│       │
│       ├── models/                  # ML & Deep Learning Services
│       │   ├── image_model.py       # MobileNetV2 predictor + Grad-CAM XAI generator
│       │   └── symptom_model.py     # Random Forest, Naive Bayes & Decision Tree models
│       │
│       ├── rag/                     # Retrieval-Augmented Generation
│       │   └── vector_store.py      # TF-IDF / FAISS veterinary knowledge store
│       │
│       └── database/                # Relational Database Models
│           └── models.py            # Animals, Cases, Symptoms, Predictions, Sessions
│
├── frontend/                        # Modern React UI
│   ├── public/
│   │   └── index.html               # HTML entry point
│   ├── src/
│   │   ├── App.js                   # Main Clinical Diagnostic Dashboard
│   │   └── index.js                 # React DOM Root
│   └── package.json                 # React dependencies
│
├── data/                            # Datasets & Weights
│   ├── Training_20symptoms.csv      # Symptom classification dataset
│   └── class_names.json             # Disease label mappings
│
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions CI/CD pipeline
│
├── backend.Dockerfile               # Production Docker container for FastAPI
├── docker-compose.yml               # Multi-container orchestration (React + FastAPI)
├── requirements.txt                 # Python environment dependencies
└── README.md                        # Documentation
```

---

## ⚙️ 4. Quick Start (Running Locally)

### Prerequisites
- Python 3.10+
- Node.js 18+

### Step 1: Start Backend (FastAPI + LangGraph)
```bash
# In project root
python -m pip install -r requirements.txt
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at: **http://localhost:8000/docs**

### Step 2: Start Frontend (React UI)
```bash
# In a new terminal window
cd frontend
npm install
npm start
```
The dashboard will open automatically at: **http://localhost:3000**

---

## 🐳 5. Running with Docker Compose

To deploy both the frontend and backend with a single command:
```bash
docker compose up --build
```

---

## 🔬 6. Core Subsystems Explained

### A. Machine Learning Symptom Ensemble
- Trains three distinct supervised models on `Training_20symptoms.csv`:
  - **Random Forest Classifier** (`n_estimators=100`)
  - **Gaussian Naive Bayes**
  - **Decision Tree Classifier**
- Performs majority voting and reports probabilistic agreement across models.

### B. Deep Learning & Explainable AI (Grad-CAM)
- Uses **MobileNetV2** transfer learning.
- Extracts spatial gradient activations from the final convolutional layer using TensorFlow's `GradientTape`.
- Superimposes a color heatmap onto the cattle image to pinpoint visual indicators (e.g. skin nodules, udder swelling, mouth lesions).

### C. RAG (Retrieval-Augmented Generation)
- Embeds verified veterinary clinical information covering Foot & Mouth Disease, Lumpy Skin Disease, Mastitis, Blackleg, Foot Rot, and Bovine Diarrhea.
- Searches and injects factual clinical chunks into the agent context to prevent LLM hallucinations.

### D. Multimodal Decision Fusion
- Correlates visual findings with reported clinical symptoms.
- Uses Generative AI (Gemini 2.5 Pro) when an API key is provided, or a deterministic rule-based fusion fallback to guarantee 100% uptime and reliable diagnosis.

---

## 🛡️ 7. AI Safety & Clinical Disclaimer
This system provides **AI-assisted preliminary cattle health assessment** and is strictly designed to assist farmers and livestock managers. It does not replace a physical clinical examination by a licensed veterinary practitioner.
