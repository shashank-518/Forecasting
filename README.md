# 📈 State-Wise Sales Forecasting System

### Production-Ready Time Series Forecasting with FastAPI • Streamlit

---

## 🚀 Overview

This project is a **production-grade Machine Learning forecasting system** that predicts the **next 8 weeks of beverage sales for each US state** using multiple time-series and machine learning models.

Unlike typical notebook-based projects, this system is built using **real-world ML engineering practices**:

- Modular pipeline architecture
- Automatic best-model selection
- Per-state model storage
- FastAPI backend for inference
- Streamlit dashboard for visualization


The system is designed to be **scalable, fast, and production-ready**.

---

## 🎯 Problem Statement

Given historical beverage sales data across multiple states:

- Irregular time intervals
- Independent state behavior
- Weekly forecasting requirement
- Need fast inference

### Goal

Predict **future 8 weeks sales per state** while:

- Maximizing accuracy
- Minimizing latency
- Supporting real-time predictions

---

## 🧠 Solution Approach

Each state is treated as an **independent time series**.

For every state:

1. Clean & preprocess data
2. Resample weekly
3. Generate lag features
4. Train multiple models
5. Evaluate metrics
6. Select best model automatically
7. Save model as pickle
8. Serve predictions via API

---

## 🤖 Models Implemented

 - ARIMA 
-  SARIMA 
 - Prophet
 - XGBoost 
 - LSTM 

### Model Selection Criteria

- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)

Lowest error model is automatically selected.

---

## ⚙️ Feature Engineering

The system automatically creates:

- Lag features (t-1, t-7, t-30)
- Rolling mean
- Rolling standard deviation
- Weekly resampling
- Time-based train/test split (no leakage)

---

## 🏗️ System Architecture

```
Training Pipeline
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
Pickle per State
      ↓
FastAPI Backend (Inference)
      ↓
Streamlit Dashboard (Frontend)
```

---

## 📂 Project Structure

```
src/
└── forecasting/
    ├── components/
    │   ├── data_ingestion.py
    │   ├── data_validation.py
    │   ├── preprocessing.py
    │   ├── feature_engineering.py
    │   ├── model_trainer.py
    │   ├── model_evaluator.py
    │   └── model_selector.py
    └── pipeline/
        ├── training_pipeline.py
        └── prediction_pipeline.py

artifacts/
└── models/

app.py
streamlit_app.py
main.py
Dockerfile
requirements.txt
README.md
```

---

## 🚀 Local Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python main.py
```

This generates:

```bash
artifacts/models/*.pkl
best_model_per_state.csv
```

### 3. Start Backend

```bash
python -m uvicorn app:app --reload
```

### 4. Start Frontend

```bash
streamlit run streamlit_app.py
```

---

## 🔌 API Usage

### Endpoint

```
GET /forecast/{state}
```

### Example Request

```bash
GET /forecast/Texas
```

### Sample Response

```json
{
  "state": "Texas",
  "model_used": "XGBoost",
  "forecast": [
    {
      "date": "10-12-2023",
      "forecast": 920000000
    }
  ]
}
```

---

## 📊 Model Training Results

The system automatically selects the best-performing model for each state based on evaluation metrics. Results are saved in `best_model_per_state.csv`.

---



## 🛠️ Technologies Used

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **ML/DL**: scikit-learn, XGBoost, TensorFlow/Keras, Prophet, statsmodels
- **Data Processing**: pandas, numpy
- **Deployment**: Docker

---


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or feedback, please open an issue in the repository.