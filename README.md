# 🩺 AI Diabetes Detection Platform

> **Production-Ready Machine Learning Pipeline** for Binary Diabetes Classification  
> Built with Random Forest, optimized via GridSearchCV, and deployable as a SaaS microservice.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Artifacts & Deployment](#artifacts--deployment)
- [API Integration](#api-integration)
- [Medical AI Considerations](#medical-ai-considerations)
- [Results & Performance](#results--performance)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

This project implements a **complete ML lifecycle** for diabetes detection using the BRFSS Diabetes Health Indicators dataset. The pipeline is designed for:

| Goal | Description |
|------|-------------|
| **PFE Submission** | Academic-grade notebook with full methodology documentation |
| **Startup Prototype** | Production-ready artifacts for immediate deployment |
| **SaaS Integration** | FastAPI-compatible inference pipeline with partial feature support |

### Key Features

- ✅ **Auto-Detection** — Automatically loads `dataset.arrow` (or `diabetes.csv` fallback)
- ✅ **Leakage-Free** — Train/test split before any preprocessing
- ✅ **Conditional SMOTE** — Applied only when imbalance exceeds 60/40 threshold
- ✅ **Conditional PCA** — Applied only when feature redundancy is detected
- ✅ **Conditional Scaling** — Automatically chooses StandardScaler or RobustScaler
- ✅ **GridSearchCV Optimization** — Exhaustive hyperparameter tuning with stratified CV
- ✅ **Deployable Artifacts** — `.pkl` models + JSON metadata for backend integration
- ✅ **Partial Feature Input** — Prediction function handles missing features gracefully

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI DIABETES DETECTION PLATFORM                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐    │
│  │  Data Layer  │──▶│  ML Pipeline │──▶│  Artifact Store   │    │
│  │  (CSV/Arrow) │   │  (Notebook)  │   │  (PKL/JSON)       │    │
│  └─────────────┘   └──────────────┘   └───────────────────┘    │
│                                               │                 │
│                                               ▼                 │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐    │
│  │  Frontend    │◀─▶│  FastAPI     │◀─▶│  Inference        │    │
│  │  (Flutter)   │   │  Backend     │   │  Service          │    │
│  └─────────────┘   └──────────────┘   └───────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

**Source**: [BRFSS Diabetes Health Indicators](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

| Property | Value |
|----------|-------|
| **Rows** | ~70,000+ |
| **Features** | 21 health indicators |
| **Target** | `Diabetes_binary` (0 = No Diabetes, 1 = Diabetes) |
| **Format** | Apache Arrow (`dataset.arrow`) with IPC Stream parsing |

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `HighBP` | Binary | High Blood Pressure (0/1) |
| `HighChol` | Binary | High Cholesterol (0/1) |
| `CholCheck` | Binary | Cholesterol Check in past 5 years (0/1) |
| `BMI` | Continuous | Body Mass Index |
| `Smoker` | Binary | Smoked at least 100 cigarettes (0/1) |
| `Stroke` | Binary | Ever had a stroke (0/1) |
| `HeartDiseaseorAttack` | Binary | Coronary heart disease or MI (0/1) |
| `PhysActivity` | Binary | Physical activity in past 30 days (0/1) |
| `Fruits` | Binary | Consume fruit 1+ times per day (0/1) |
| `Veggies` | Binary | Consume vegetables 1+ times per day (0/1) |
| `HvyAlcoholConsump` | Binary | Heavy alcohol consumption (0/1) |
| `AnyHealthcare` | Binary | Has any healthcare coverage (0/1) |
| `NoDocbcCost` | Binary | Could not see doctor due to cost (0/1) |
| `GenHlth` | Ordinal | General health (1=Excellent → 5=Poor) |
| `MentHlth` | Continuous | Days of poor mental health (0–30) |
| `PhysHlth` | Continuous | Days of poor physical health (0–30) |
| `DiffWalk` | Binary | Difficulty walking or climbing stairs (0/1) |
| `Sex` | Binary | Sex (0=Female, 1=Male) |
| `Age` | Ordinal | Age category (1–13) |
| `Education` | Ordinal | Education level (1–6) |
| `Income` | Ordinal | Income level (1–8) |

---

## 🔬 ML Pipeline

The notebook follows a **strict 11-step production pipeline**:

```
Step 1  ─ Import Libraries & Global Config
Step 2  ─ Load Dataset (Auto-Detection)
Step 3  ─ Train/Test Split (Stratified, 80/20)
Step 4  ─ Feature Processing Pipeline (Reusable)
Step 4.5 ─ Apply Feature Pipeline
Step 5  ─ Conditional SMOTE
Step 6  ─ Conditional Normalization
Step 7  ─ Conditional PCA
Step 7.5 ─ Feature & PCA Analysis
Step 8  ─ Train & Optimize Random Forest (GridSearchCV)
Step 9  ─ Comprehensive Evaluation
Step 10 ─ Save Model Artifacts
Step 11 ─ Deployment-Ready Prediction Function
```

### Pipeline Design Principles

| Principle | Implementation |
|-----------|---------------|
| **No Data Leakage** | Split happens before ALL preprocessing |
| **Reproducibility** | `random_state=42` applied everywhere |
| **Conditional Logic** | SMOTE, PCA, and scaling are auto-decided |
| **Reusability** | `FeaturePipeline` class is serializable |
| **Deployability** | All artifacts saved for backend inference |

---

## 📁 Project Structure

```
ML/
├── data/
│   ├── diabetes.csv          # Primary dataset (CSV format)
│   └── diabetes.arrow        # Alternative dataset (Arrow format)
├── artifacts/                # Generated after running notebook
│   ├── rf_model.pkl          # Trained Random Forest model
│   ├── scaler.pkl            # Fitted scaler (if applied)
│   ├── pca.pkl               # Fitted PCA (if applied)
│   ├── feature_pipeline.pkl  # Reusable feature processor
│   └── feature_list.json     # Feature names + model metadata
├── diabetes.ipynb            # ⭐ Main ML notebook
├── requirements.txt          # Python dependencies
├── generate_notebook.py      # Notebook generation script
└── README.md                 # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. **Clone the repository** (or download the workspace):
   ```bash
   cd ML/
   ```

2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1    # PowerShell
   # or
   .\.venv\Scripts\activate        # CMD
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Place your dataset** in the `data/` directory as either:
   - `data/diabetes.csv`
   - `data/diabetes.arrow`

5. **Run the notebook**:
   ```bash
   jupyter notebook diabetes.ipynb
   ```
   Or use VS Code / JupyterLab to open and run all cells.

---

## 📓 Notebook Walkthrough

### Step 1 — Import Libraries
All necessary libraries are imported with clean organization. Global configuration sets:
- `RANDOM_STATE = 42` for reproducibility
- Professional plotting style via seaborn

### Step 2 — Load Dataset
Auto-detection logic checks for `diabetes.csv` → `diabetes.arrow` → any CSV/Arrow fallback. Target column is automatically identified.

### Step 3 — Train/Test Split
Stratified 80/20 split **before any preprocessing** to prevent data leakage. Class distribution is validated.

### Step 4 & 4.5 — Feature Processing
A reusable `FeaturePipeline` class handles:
- Missing value imputation (training medians)
- IQR-based outlier clipping
- Feature engineering (BMI Risk Category, Health Score)
- Pipeline is fitted on train data, applied to both sets

### Step 5 — Conditional SMOTE
SMOTE is applied **only if** majority class > 60%. Applied **only to training data**.

### Step 6 — Conditional Normalization
Automatically selects `RobustScaler` (high skewness) or `StandardScaler` (moderate). For Random Forest, scaling is optional but included for pipeline portability.

### Step 7 & 7.5 — Conditional PCA & Feature Analysis
PCA is applied only when high correlation pairs (r > 0.85) exceed a threshold. Feature importance and correlation heatmaps identify the most decisive medical indicators.

### Step 8 — Model Training
`RandomForestClassifier` + `GridSearchCV` with:
- 5-fold stratified cross-validation
- F1-score optimization
- Hyperparameter grid: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight`

### Step 9 — Evaluation
Comprehensive metrics on the held-out test set:
- Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix, ROC curve, Feature importance chart
- Bias analysis with false negative rate discussion

### Step 10 — Save Artifacts
All components serialized to `artifacts/` for deployment:
- Model, scaler, PCA, feature pipeline, metadata JSON

### Step 11 — Prediction Function
`predict_diabetes(input_dict)` — a deployment-ready function that:
- Accepts partial features
- Fills missing values safely
- Returns `{prediction, probability, risk_level}`

---

## 📦 Artifacts & Deployment

After running the notebook, the `artifacts/` directory contains:

| File | Description |
|------|-------------|
| `rf_model.pkl` | Trained and optimized Random Forest model |
| `scaler.pkl` | Fitted scaler (StandardScaler or RobustScaler) |
| `pca.pkl` | Fitted PCA transformer (if applied) |
| `feature_pipeline.pkl` | Reusable FeaturePipeline instance |
| `feature_list.json` | Feature names, metadata, best params, test metrics |

### `feature_list.json` Structure
```json
{
  "feature_names": ["HighBP", "HighChol", "BMI", ...],
  "target_column": "Diabetes_binary",
  "scaler_used": "StandardScaler",
  "pca_applied": false,
  "smote_applied": true,
  "best_params": {"n_estimators": 300, "max_depth": 20, ...},
  "test_metrics": {"Accuracy": 0.75, "F1-Score": 0.73, ...},
  "random_state": 42
}
```

---

## 🌐 API Integration

### FastAPI Backend Example

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import pandas as pd

app = FastAPI(title="Diabetes Detection API", version="1.0.0")

# Load artifacts at startup
model = joblib.load("artifacts/rf_model.pkl")
pipeline = joblib.load("artifacts/feature_pipeline.pkl")
metadata = json.load(open("artifacts/feature_list.json"))

class PredictionRequest(BaseModel):
    HighBP: float = None
    HighChol: float = None
    BMI: float = None
    Smoker: float = None
    GenHlth: float = None
    Age: float = None
    # ... other optional features

@app.post("/predict")
async def predict(request: PredictionRequest):
    input_dict = {k: v for k, v in request.dict().items() if v is not None}
    
    # Build input with defaults for missing features
    row = {}
    for feat in metadata["feature_names"]:
        row[feat] = input_dict.get(feat, pipeline.medians.get(feat, 0))
    
    X = pd.DataFrame([row])
    X = pipeline.transform(X)
    
    # Apply scaler & PCA if used during training
    # ... (see notebook Step 11 for full logic)
    
    prediction = int(model.predict(X.values)[0])
    probability = float(model.predict_proba(X.values)[0][1])
    
    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_level": "Low" if probability < 0.3 else "Medium" if probability < 0.6 else "High"
    }
```

### Run the API

```bash
pip install fastapi uvicorn
uvicorn api:app --reload --port 8000
```

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"HighBP": 1, "HighChol": 1, "BMI": 35, "GenHlth": 4, "Age": 10}'
```

### Example Response

```json
{
  "prediction": 1,
  "probability": 0.7234,
  "risk_level": "High"
}
```

---

## ⚕️ Medical AI Considerations

### Bias & Fairness

| Concern | Mitigation |
|---------|-----------|
| **Class Imbalance** | Conditional SMOTE + balanced class weights |
| **Feature Bias** | Feature importance analysis identifies dominant predictors |
| **Demographic Bias** | Sex, Age, Income included — monitor for disparate impact |
| **False Negatives** | Critical metric — missed diabetics have real health consequences |

### False Negative Analysis

In healthcare AI, **false negatives** (missed positive cases) are more dangerous than false positives:

- A missed diabetic patient won't receive early intervention
- This can lead to severe complications (neuropathy, retinopathy, kidney disease)
- **Recall** is therefore prioritized alongside F1-score in model evaluation

### Model Explainability

- **Feature importance** rankings provide clinical interpretability
- **BMI, General Health, Age, and Blood Pressure** are typically the most predictive
- The model supports **partial feature input** — clinics with limited data can still get predictions

### Regulatory Notes

> ⚠️ This model is a **screening tool**, not a diagnostic device.  
> It should complement (not replace) clinical laboratory tests (HbA1c, fasting glucose).  
> Deployment in clinical settings requires regulatory approval (FDA/CE marking).

---

## 📈 Results & Performance

### Actual Performance (dataset.arrow)

| Metric | Score Achieved |
|--------|---------------|
| **Accuracy** | 74.88% |
| **ROC-AUC** | 82.45% |
| **Recall** | 79.43% |
| **Precision** | 72.80% |
| **F1-Score** | 75.97% |

> The final optimized Random Forest parameters chosen by GridSearchCV on the Apache Arrow dataset were `max_depth = 10`, `n_estimators = 200`, `min_samples_split = 5`, and `class_weight = 'balanced_subsample'`.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.9+ |
| **ML Framework** | scikit-learn |
| **Imbalanced Learning** | imbalanced-learn (SMOTE) |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Serialization** | joblib |
| **Notebook** | Jupyter / VS Code |
| **Backend (planned)** | FastAPI |
| **Frontend (planned)** | Flutter |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is developed as part of a **PFE (Projet de Fin d'Études)** and startup prototype.  
All rights reserved © 2026.

---

<p align="center">
  <strong>Built with ❤️ for Medical AI</strong><br>
  <em>Senior ML Engineering • FAANG-Level Pipeline • SaaS-Ready</em>
</p>
