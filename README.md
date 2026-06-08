<div align='center'>
    
# Titanic Survival Prediction

![Docker CI](https://github.com/Stewie-pixel/titanic-survival-prediction/actions/workflows/docker.yml/badge.svg)
[![Dataset](https://img.shields.io/badge/Dataset-891%20rows-blue?logo=kaggle)](https://www.kaggle.com/competitions/titanic)
[![Ensemble Accuracy](https://img.shields.io/badge/Ensemble%20Accuracy-%3E82%25-success?logo=python)](https://github.com/Stewie-pixel/titanic-survival-prediction)

A machine learning pipeline to predict Titanic passenger survival using feature engineering and a Voting Ensemble of Random Forest, Gradient Boosting and XGBoost, deployed as a FastAPI service on Hugging Face Spaces.

</div>

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Pipeline](#pipeline)
5. [Models](#models)
6. [Getting Started](#getting-started)
7. [API Usage](#api-usage)
8. [CI/CD](#cicd)
9. [Tech Stack](#tech-stack)

## Overview

This project builds an end-to-end survival prediction pipeline covering data exploration, preprocessing, feature engineering, model training, hyperparameter tuning and deployment. Three models are trained and compared — Random Forest, Gradient Boosting and XGBoost — before being combined into a soft Voting Ensemble for maximum accuracy.

## Dataset
 
| Property | Value |
|---|---|
| Source | Kaggle — Titanic: Machine Learning from Disaster |
| Train rows | 891 |
| Test rows | 418 |
| Columns | 12 |
| Target | `Survived` (0 = Died, 1 = Survived) |
 
**Features:**
 
| Feature | Type | Description |
|---|---|---|
| `Pclass` | Numerical | Passenger class (1st, 2nd, 3rd) |
| `Sex` | Categorical | Male / Female |
| `Age` | Numerical | Passenger age (imputed by Title group median) |
| `SibSp` | Numerical | Number of siblings / spouses aboard |
| `Parch` | Numerical | Number of parents / children aboard |
| `Fare` | Numerical | Ticket fare |
| `Embarked` | Categorical | Port of embarkation (C, Q, S) |
| `Cabin` | Categorical | Cabin number (deck extracted) |

## Project Structure

```
titanic-survival-prediction/
├── .github/
│   └── workflows/
│       └── deploy.yml
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│   └── submission.csv
├── models/
│   ├── model.pkl
│   ├── encoders.pkl
│   ├── feature_names.pkl
│   └── metrics.pkl
├── src/
│   └── kaggle/
│       └── submission.csv
│   └── titanic_model.ipynb
├── main.py
├── model.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
└── README.md
```

## Pipeline

The full pipeline is documented in `src/titanic_model.ipynb` across 10 sections:
 
1. **Import Libraries & Load Data** — load and inspect the dataset
2. **Data Overview & Initial Inspection** — shape, dtypes, missing values
3. **Exploratory Data Analysis** — survival distribution, numerical and categorical analysis, correlation
4. **Data Preprocessing & Cleaning** — handle missing values, drop low-signal columns
5. **Feature Engineering** — derive new features to improve model performance
6. **Modelling** — train Random Forest, Gradient Boosting and XGBoost individually
7. **Hyperparameter Tuning** — GridSearchCV on all three models
8. **Voting Ensemble** — combine tuned models with soft voting
9. **Evaluation** — confusion matrix, cross validation, feature importance, ROC curve
10. **Save Pipeline Artifacts** — export model and encoders to `/models` as `.pkl` files
**Engineered Features:**
 
| Feature | Description |
|---|---|
| `Title` | Extracted from Name (Mr, Mrs, Miss, Master, Rare) |
| `FamilySize` | SibSp + Parch + 1 |
| `FamilyGroup` | Alone / Small / Large |
| `IsAlone` | 1 if travelling alone |
| `CabinDeck` | First letter of Cabin, Unknown if missing |
| `AgeBand` | Age binned into Child / Teen / Adult / MidAge / Senior |
| `FareBand` | Fare binned into Low / Medium / High / Very High quartiles |
| `FarePerPerson` | Fare / FamilySize |
| `AgeClass` | Age × Pclass interaction |

## Models
 
### Random Forest
Ensemble of decision trees using majority voting. Strong baseline, resistant to overfitting with tuned `max_depth` and `min_samples_leaf`.
 
### Gradient Boosting
Sequential tree ensemble that corrects previous errors. Captures complex non-linear patterns with tuned `learning_rate` and `subsample`.
 
### XGBoost
Optimised gradient boosting implementation. Fastest convergence with tuned `colsample_bytree` and `subsample`.
 
### Voting Ensemble
Soft voting ensemble combining all three models. Final prediction is the average of predicted probabilities across all classifiers.
 
### Model Comparison
 
| Model | CV Accuracy | CV Std |
|---|---|---|
| Random Forest | — | — |
| Gradient Boosting | — | — |
| XGBoost | — | — |
| **Voting Ensemble** | **—** | **—** |
 
> Update this table with actual values after running the notebook.
 
 
## Getting Started
 
**Prerequisites:**
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
**Install dependencies:**
```bash
uv sync
```
 
**Run the full notebook:**
 
Open `src/titanic_model.ipynb` in VS Code with the Jupyter extension and run all cells via `Kernel → Restart & Run All`.
 
 
## API Usage
 
The model is deployed as a FastAPI service on Hugging Face Spaces.
 
**Base URL:**
```
https://stewie-pixel-titanic-survival-prediction.hf.space
```
 
**Endpoints:**
 
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/metrics` | GET | Model CV accuracy and metrics |
| `/predict` | POST | Predict survival |
 
**Example request:**
```bash
curl -X POST https://stewie-pixel-titanic-survival-prediction.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 29,
    "sibsp": 0,
    "parch": 0,
    "fare": 211.34,
    "embarked": "S"
  }'
```
 
**Example response:**
```json
{
  "input": { ... },
  "prediction": 1,
  "survival_probability": 0.94,
  "label": "Survived"
}
```
 
**Interactive docs:**
```
https://stewie-pixel-titanic-survival-prediction.hf.space/docs
```
 
 
## CI/CD
 
This project uses GitHub Actions for continuous deployment to Hugging Face Spaces.
 
**Workflow: `.github/workflows/deploy.yml`**
 
| Trigger | Action |
|---|---|
| Push to `main` | Upload to HF Space → Dispatch to AI-project-collection |
| Pull Request to `main` | No deployment |
 
**Pipeline steps:**
1. Checkout repository
2. Install `huggingface_hub` with Xet support
3. Upload app files and model artifacts to HF Space via Xet storage
4. Trigger `deploy-single.yml` in `AI-project-collection` to register model
**Required GitHub Secrets:**
 
| Secret | Description |
|---|---|
| `HF_TOKEN` | Hugging Face write access token |
| `CENTRAL_REPO_TOKEN` | GitHub PAT with `repo` scope |
 
 
## Tech Stack
 
| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| pandas | Data manipulation |
| numpy | Numerical computing |
| scikit-learn | Preprocessing, RF, GB, GridSearchCV |
| XGBoost | Gradient boosting model |
| FastAPI | REST API framework |
| uvicorn | ASGI server |
| seaborn / matplotlib | Data visualization |
| Jupyter Notebook | Exploratory pipeline |
| Hugging Face Spaces | Model hosting |
| GitHub Actions | CI/CD pipeline |
| uv | Package management |
| VS Code | Development environment |