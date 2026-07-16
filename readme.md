# Heart Disease Prediction System Using Machine Learning

## Project Overview

This project was developed as part of the Master's programme at the National College of Ireland. The aim is to design and implement a machine learning application capable of predicting different cardiovascular conditions using three independent healthcare datasets.

Rather than relying on a single dataset, this project compares multiple cardiovascular datasets, evaluates several machine learning algorithms, and deploys the most suitable model for each prediction task within a unified Flask web application.

The completed system allows users to perform predictions for:

- Ten-Year Coronary Heart Disease Risk (Framingham Dataset)
- Heart Disease Detection (Heart Disease Dataset)
- Heart Failure Outcome Prediction (Heart Failure Clinical Records Dataset)

The project also includes a comparison dashboard that summarises the datasets, deployed models, and evaluation metrics.

---

# Project Objectives

The main objectives of this project are:

- Develop machine learning models for cardiovascular risk prediction.
- Compare multiple publicly available healthcare datasets.
- Evaluate different machine learning algorithms.
- Deploy the selected models using Flask.
- Provide an interactive web application for educational demonstration.
- Present comparative model performance using evaluation metrics.

---

# Application Features

The application contains five main modules.

### Framingham Prediction Module

Predicts the probability of developing coronary heart disease over a ten-year period using demographic, behavioural and clinical information.

Selected deployment model:

- Random Forest

---

### Heart Disease Detection Module

Predicts whether a patient is likely to have heart disease using clinical measurements obtained from the Heart Disease dataset.

Selected deployment model:

- XGBoost

---

### Heart Failure Prediction Module

Predicts adverse heart failure outcomes using patient clinical measurements collected in the Heart Failure Clinical Records dataset.

Selected deployment model:

- XGBoost

---

### Model Comparison Dashboard

The dashboard provides a comparison of:

- Datasets
- Selected machine learning models
- Accuracy
- Precision
- Recall
- F1-score
- Overall model observations

---

### About Page

Provides project background, technologies used, workflow and important academic information.

---

# Datasets

## 1. Framingham Heart Study Dataset

Purpose:

Prediction of ten-year coronary heart disease risk.

Characteristics:

- Approximately 4,240 patient records
- 15 input features
- Binary target variable

---

## 2. Heart Disease Dataset

Purpose:

Prediction of heart disease presence.

Characteristics:

- Approximately 920 patient records
- 12 selected input features after preprocessing
- Binary classification

---

## 3. Heart Failure Clinical Records Dataset

Purpose:

Prediction of heart failure outcomes.

Characteristics:

- Approximately 5,000 patient records
- 12 clinical variables
- Binary target variable

---

# Machine Learning Algorithms Evaluated

Multiple algorithms were investigated during model development.

These included:

- Logistic Regression
- Random Forest
- Random Forest with SMOTE
- XGBoost

Each model was evaluated using standard classification metrics before selecting the final deployment model.

---

# Model Performance

| Dataset | Selected Model | Accuracy | Precision | Recall | F1 Score |
|----------|---------------|----------|-----------|--------|----------|
| Framingham | Random Forest | 81.60% | 0.315 | 0.178 | 0.228 |
| Heart Dataset | XGBoost | 83.70% | 0.846 | 0.863 | 0.854 |
| Heart Failure | XGBoost | 99.50% | 0.994 | 0.990 | 0.992 |

The evaluation metrics shown above correspond to the models deployed within the Flask application.

---

# Technologies Used

Programming Language

- Python

Machine Learning

- Scikit-learn
- XGBoost
- Pandas
- NumPy

Web Framework

- Flask

Frontend

- HTML5
- CSS3

Development Tools

- Visual Studio Code
- Git
- GitHub

---

# Project Structure

```
HEART_DISEASE_PREDICTION_PROJECT

│
├── app
│   ├── app.py
│   ├── models
│   ├── static
│   └── templates
│
├── datasets
├── images
├── notebooks
├── reports
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Running the Application

Clone the repository

```bash
git clone <https://github.com/Manohar539/Heart-Disease-Prediction.git>
```

Move into the project directory

```bash
cd HEART_DISEASE_PREDICTION_PROJECT
```

Install the required packages

```bash
pip install -r requirements.txt
```

Start the Flask application

```bash
cd app
python app.py
```

Open the application

```
http://127.0.0.1:5000
```

---

# Application Workflow

The overall workflow followed in this project is:

```
Dataset Collection
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Model Selection
        │
        ▼
Flask Integration
        │
        ▼
Risk Prediction
```

---

# Educational Purpose

This application has been developed as part of a Master's degree project.

The prediction results are intended to demonstrate the application of machine learning techniques in healthcare datasets.

The system is not designed for clinical diagnosis and should not replace professional medical advice.

---

# Future Improvements

Several enhancements could be considered in future work:

- Deploy the application on a cloud platform such as AWS or Azure.
- Integrate additional cardiovascular datasets.
- Support authenticated user accounts.
- Store prediction history in a database.
- Include explainable AI techniques such as SHAP or LIME to improve model interpretability.
- Develop REST APIs for external integration.

---

# Author

**Master's Project**

National College of Ireland

Machine Learning Module

Academic Year: 2025–2026

---

# Acknowledgements

The datasets used in this project are publicly available for academic research. This work has been completed for educational purposes as part of postgraduate study at the National College of Ireland.