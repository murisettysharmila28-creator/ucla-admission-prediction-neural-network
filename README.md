# UCLA Admission Prediction using Neural Network

## Live App

https://ucla-admission-prediction-neural-network-sharmila.streamlit.app/

---

## Overview

This project demonstrates an end-to-end machine learning pipeline for predicting whether an applicant is likely to have a strong admission chance using a neural network model.

The solution includes:
- data preprocessing
- target transformation
- neural network model training
- model evaluation
- deployment as an interactive Streamlit web application

The original admission probability target was converted into a binary classification problem, where applicants with an admission chance of 0.8 or higher are treated as having a strong admission chance.

---

## Problem Statement

Graduate admission datasets often provide a probability-like admission score rather than a direct class label. This project reformulates the problem into a binary classification task to determine whether a candidate is likely to belong to the high-admission-chance group.

The goal is to use academic and profile-related features to predict whether an applicant is likely to have a strong admission chance.

---

## Dataset

- Dataset: Admission.csv
- Target column: `Admit_Chance`
- Target transformation:
  - `1` if `Admit_Chance >= 0.8`
  - `0` otherwise

### Features Used

- GRE_Score
- TOEFL_Score
- University_Rating
- SOP
- LOR
- CGPA
- Research

### Dropped Column

- Serial_No

---

## Approach

### 1. Data Preparation
- Removed the `Serial_No` column
- Converted the original admission score into a binary target
- Treated `University_Rating` as a categorical feature
- Scaled numeric features using `StandardScaler`
- Applied one-hot encoding to categorical features

### 2. Model Training
- Algorithm used: `MLPClassifier`
- Hidden layers: `(32, 16)`
- Activation: `relu`
- Optimizer: `adam`

### 3. Evaluation
The model was evaluated using classification accuracy.

### Results
- Train Accuracy: **0.9950**
- Test Accuracy: **0.8300**

---

## Application Features

The Streamlit app allows users to:
- enter applicant profile details
- predict whether the applicant is likely to have a strong admission chance
- view the model-estimated probability
- read a short interpretation of the result

---

## How the Prediction Works

The deployed app predicts whether an applicant belongs to the **high-admission-chance group**, not whether they are guaranteed admission.

This means the output should be interpreted as:
- **Likely to Have a Strong Admission Chance**
- **Unlikely to Have a Strong Admission Chance**

rather than a direct real-world admission decision.

---

## Project Structure

```bash
ucla-admission-prediction-neural-network/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Admission.csv
│
├── model/
│   ├── admission_model.pkl
│   ├── preprocessor.pkl
│   └── feature_columns.pkl
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── logger.py
│
└── notebooks/
    └── UCLA_Neural_Networks_Solution.ipynb

```
## Installation

Clone the repository:

```bash

git clone https://github.com/your-username/ucla-admission-prediction-neural-network.git
cd ucla-admission-prediction-neural-network

Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate

Install Dependencies
pip install -r requirements.txt


## Run the Project

1. Train the model
python main.py

2. Run the Streamlit app
python -m streamlit run app.py

```

## Tech Stack

Python
Pandas
Scikit-learn
Streamlit
Joblib

---

## Key Learnings

- Converting a continuous target into a binary classification task
- Building preprocessing pipelines with scaling and categorical encoding
- Training and evaluating neural network classifiers
- Separating training, preprocessing, and inference into modular components
- Deploying a machine learning model as an interactive web application

---

## Limitations

- The model predicts membership in a high-admission-chance group, not actual admission decisions
- Probabilities from neural networks may not always be perfectly calibrated
- The dataset is relatively small, so generalization may be limited

## Future Improvements

- Add probability calibration for more reliable probability estimates
- Compare performance with logistic regression, random forest, and gradient boosting
- Add feature importance or explainability methods
- Improve UI/UX and applicant profile summaries
---

## Author

Sharmila Murisetty
Graduate Student – Business Intelligence & Systems Infrastructure
Aspiring Data Analyst / BI Developer