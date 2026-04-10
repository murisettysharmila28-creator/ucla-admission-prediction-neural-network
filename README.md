# UCLA Admission Prediction using Neural Network

## Live App

https://ucla-admission-prediction-neural-network-sharmila.streamlit.app/

---

## Overview

This project demonstrates an end-to-end machine learning pipeline for predicting whether an applicant is likely to have a strong admission chance using a neural network model.

The solution includes:
- data preprocessing
- target transformation
- baseline model training
- hyperparameter tuning using GridSearchCV
- model evaluation and validation
- deployment as an interactive Streamlit web application

The original admission probability target was converted into a binary classification problem, where applicants with an admission chance of 0.8 or higher are treated as having a strong admission chance.

---

## Problem Statement

Graduate admission datasets often provide a probability-like admission score rather than a direct class label. This project reformulates the problem into a binary classification task to determine whether a candidate is likely to belong to the high-admission-chance group.

The goal is to use academic and profile-related features to predict whether an applicant is likely to have a strong admission chance.

---

## Dataset

- Dataset: Admission.csv
- Target column: Admit_Chance

### Target Transformation
- 1 if Admit_Chance >= 0.8
- 0 otherwise

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
- Removed the Serial_No column
- Converted the original admission score into a binary target
- Treated University_Rating as a categorical feature
- Scaled numeric features using StandardScaler
- Applied one-hot encoding using OneHotEncoder

### 2. Baseline Model
- Algorithm: MLPClassifier
- Hidden layers: (32, 16)
- Activation: relu
- Early stopping enabled

### Baseline Results
- Train Accuracy: 0.9050
- Test Accuracy: 0.8200

### 3. Hyperparameter Tuning
- Used GridSearchCV with 5-fold cross-validation
- Tuned:
  - hidden layer sizes
  - activation function
  - regularization (alpha)
  - learning rate
  - max iterations

### Best Hyperparameters
{
 'activation': 'tanh',
 'alpha': 0.0001,
 'hidden_layer_sizes': (64, 32),
 'learning_rate_init': 0.01,
 'max_iter': 1000,
 'early_stopping': True,
 'solver': 'adam'
}

### 4. Final Model Performance

- Train Accuracy: 0.9425
- Test Accuracy: 0.9000
- Precision: 0.8000
- Recall: 0.9032
- F1 Score: 0.8485

### Cross-Validation Results
- Mean Accuracy: 0.9120
- Standard Deviation: 0.0271

---

## Model Interpretation

The tuned model shows:
- strong overall accuracy
- balanced precision and recall
- improved generalization compared to baseline

---

## Application Features

The Streamlit app allows users to:
- input applicant profile details
- predict admission likelihood
- view prediction probability
- understand results in simple terms

---

## How the Prediction Works

The model predicts whether an applicant belongs to the high-admission-chance group, not guaranteed admission.

Output:
- Likely to Have a Strong Admission Chance
- Unlikely to Have a Strong Admission Chance

---

## Project Structure

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
│   ├── validation.py
│   ├── predict.py
│   └── logger.py
│
└── notebooks/
    └── UCLA_Neural_Networks_Solution.ipynb

---

## Installation

Clone the repository:

git clone https://github.com/murisettysharmila28-creator/ucla-admission-prediction-neural-network.git
cd ucla-admission-prediction-neural-network

Create virtual environment:

python -m venv venv
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

---

## Run the Project

Train the model:

python main.py

Run the Streamlit app:

python -m streamlit run app.py

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

---

## Key Learnings

- Converting regression output into classification
- Building modular ML pipelines
- Feature scaling and encoding
- Neural network model training
- Hyperparameter tuning using GridSearchCV
- Model evaluation using precision, recall, F1-score and cross-validation
- Deploying ML apps using Streamlit

---

## Limitations

- Predicts probability group, not actual admission decision
- Neural network probabilities may not be perfectly calibrated
- Small dataset size

---

## Future Improvements

- Add probability calibration
- Compare with Logistic Regression, Random Forest, XGBoost
- Add explainability (SHAP)
- Improve UI/UX

---

## Author

Sharmila Murisetty - Data Analyst / BI Developer