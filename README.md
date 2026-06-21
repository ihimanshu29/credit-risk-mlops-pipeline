# <h1 align="center">🏦 RiskEngine AI: Production-Ready Credit Underwriting Pipeline</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <a href="https://mlflow.org/"><img src="https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2.svg" alt="MLflow"></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-Containerized-2496ED.svg" alt="Docker"></a>
  <a href="https://render.com"><img src="https://img.shields.io/badge/Render-Live_Deployment-46E3B7.svg" alt="Render"></a>
</p>

---

## ⚡ Executive Summary & Overview

**RiskEngine AI** is an enterprise-grade, configuration-driven Machine Learning Operations (**MLOps**) production pipeline engineered to evaluate consumer credit risk and compute default probabilities. Powered by a high-performance **XGBoost** classification engine, this framework shifts machine learning away from isolated Jupyter notebooks into a resilient, **deployable, modular, and type-safe architecture**.

By processing multi-dimensional financial features—such as borrower age, historical income dynamics, loan structures, and historical credit-line lengths—the pipeline serves real-time credit decisioning via a high-throughput **FastAPI** / **Flask** web application dashboard.

---

## 🚀 Live Production Deployment

The entire inference application layer and frontend decisioning interface are fully containerized via **Docker** and actively orchestrating live workloads on **Render**.

- **🔗 Production Gateway:** [Access the Underwriting Dashboard Here](https://your-render-url-here.onrender.com)
- 💡 _Note: Infrastructure hosted on Render's standard tier may experience a brief 30–60 second cold-start latency during initial routing container spin-up._

---

## 🧠 Core Engineering Differentiators & Reflection

This system was meticulously built from the ground up to reflect modern production-grade architecture patterns, emphasizing clean separation of concerns and robust data governance:

- **Configuration-Driven Architecture:** Complete elimination of hardcoded operational parameters. Every ingestion endpoint, validation rule, data split, and model hyperparameter is dynamically declared inside centralized `config.yaml` and `params.yaml` layers.
- **Strict Type Safety & Schema Contracts:** Employs programmatic constraints using Python `@dataclass(frozen=True)` data transfer objects and strict `@ensure_annotations`. Any downstream data drifts or schema violations in `schema.yaml` instantly trigger explicit pipeline termination, neutralizing data corruption vectors before model poisoning can occur.
- **Experiment Governance via MLflow:** Built-in tracking integration with **MLflow**. Pipeline iterations automatically serialize, register, and stream operational metrics (**Accuracy**, **Precision**, **Recall**, **F1-Score**), runtime hyperparameters, and active model binaries directly to a centralized tracking dashboard.
- **Automated CI/CD Workflows:** Integrated with a **GitHub Actions** automation suite. Every codebase mutation triggers automated remote runners to enforce syntax validation, evaluate dependencies, and construct isolated test **Docker** container builds.

---

## ⚙️ Pipeline & Project Architecture

### 📊 System Execution Architecture

The underlying data and training flow operates through five fully decoupled, autonomous components:

```text
🏁 Pipeline Ingestion Trigger
       │
       ▼
 📦 [01_Data_Ingestion]      ──► Fetches remote zipped payloads & extracts 'credit_risk.csv'
       │
       ▼
 🛡️ [02_Data_Validation]     ──► Validates structure against strict schema.yaml constraints
       │
       ▼
 🔄 [03_Data_Transformation] ──► Orchestrates train-test splits & handles numerical vectors
       │
       ▼
 🚀 [04_Model_Trainer]        ──► Ingests parameters, runs XGBoost, and serializes weights
       │
       ▼
 📈 [05_Model_Evaluation]     ──► Quantifies classification metrics & streams telemetry to MLflow
```

---

## 🗂️ Complete Directory Topology

```text
credit-risk-mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ci-cd.yaml         # GitHub Actions Workflow Engine
├── artifacts/                  # Local Pipeline Versioned Storage
│   ├── data_ingestion/        # Extracted credit_risk.csv source layer
│   ├── data_validation/       # Schema evaluation compliance outputs
│   ├── data_transformation/   # Prepared Model-Ready Partitions (train/test)
│   ├── model_trainer/         # Serialized model.joblib binaries
│   └── model_evaluation/      # Local telemetry outputs (metrics.json)
├── config/
│   └── config.yaml            # Monolithic Pipeline Component Registry
├── src/
│   └── mlProject/
│       ├── components/        # Isolated Functional Execution Tasks
│       ├── config/            # Internal Configuration Management Engines
│       ├── entity/            # Strongly-Typed In-Memory Data Models
│       ├── pipeline/          # Orchestrated Sequential Stage Controllers
│       └── utils/             # High-Performance Common Core Utilities
├── static/                    # Frontend UI Presentation Assets
├── templates/                 # UI Execution Views (index.html, results.html)
├── Dockerfile                 # Multi-Stage App Deployment Container Specs
├── params.yaml                # XGBoost Model Hyperparameter Definitions
├── schema.yaml                # Core Data Validation Schema Declarations
├── requirements.txt           # Monitored Project Component Dependencies
└── wsgi.py                    # High-Performance Application Gateway
```

---

## 🛠️ Condensed Workflow Progression

### 1. Declarative Updates

Configure asset paths in `config.yaml`, tune XGBoost hyperparameters in `params.yaml`, and define columns in `schema.yaml`.

### 2. Contract Construction

Instantiate type-safe operational variables inside the internal `config_entity.py` domain.

### 3. Component Engineering

Program the core pipeline steps inside the isolated `components` catalog.

### 4. Execution Orchestration

Link execution blocks into isolated pipeline stages routed through `main.py`.

### 5. UI & API Integration

Bind predictive workflows to web endpoints within the service runner (`wsgi.py`).

### 6. Containerization & Deployment

Package the runtime dependencies via Docker and deploy to Render.

---

## 📈 Single-Line Operational Project Flow

```text
Data Ingestion ──► Structural Validation ──► Feature Transformation ──► XGBoost Optimization ──► MLflow Registration ──► API Inference Serving
```

---

## 🧩 Unified Enterprise Tech Stack

| Operational Domain      | Applied Technologies                              |
| ----------------------- | ------------------------------------------------- |
| Core Programming Engine | Python 3.10+                                      |
| Model Optimization      | XGBoost (Extreme Gradient Boosting Classifier)    |
| Data Orchestration      | Pandas, NumPy, Scikit-Learn, Joblib               |
| Experiment Governance   | MLflow Tracking Server & Model Registry           |
| Inference Framework     | Flask / FastAPI High-Performance Web Services     |
| Runtime Environment     | Anaconda / Miniconda Package Ecosystem            |
| Infrastructure & DevOps | Docker Engine, GitHub Actions CI/CD, Render Cloud |

---

## 💻 Local Setup & Development Environment

### Step 1: Environment Provisioning

```bash
conda create -n mlproj python=3.10 -y
conda activate mlproj
```

### Step 2: Dependency Synchronization

```bash
pip install -r requirements.txt
```

### Step 3: Launch Local Core Pipeline & Web Server

```bash
python main.py   # Runs entire End-to-End MLOps Pipeline
python wsgi.py   # Launches local development inference web interface
```

---

## 💡 Key Structural Highlights

### Production Architecture Blueprint

Implements structural modularity mimicking enterprise data software frameworks.

### Full-Lineage Reproducibility

Every operational run yields mathematically identical, auditable results governed via configuration locks.

### Telemetry Insights

Transparent evaluation layers tracking precision, recall curves, and model weights out-of-the-box.

### Turnkey Deployment Ready

Minimal cloud configurations required to transition from local testing environments straight to live production nodes.

---

## 🧾 License & Personal Dedication

This project is licensed under the MIT License—granting full authorization for modifications, business distribution, and private adaptation.

### A Note from the Author

This system serves as a showcase of modern MLOps principles, blending modern machine learning engineering with software craftsmanship.

If this repository helped you scale your production deployment mental models, feel free to give it a ⭐!
