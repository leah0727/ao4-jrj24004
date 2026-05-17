# ao4-jrj24004
A04: Model Interpretability (xAI)

# A04 — Explainable AI (xAI) for Loan Approval Prediction

## Overview

This project focuses on interpreting a machine learning classification model using explainable AI (xAI) techniques.

Using the Loan Prediction dataset from the sampling notebooks, a classification model was trained to predict the target variable:

- `Loan_Status`

The goal of this assignment was not to maximize predictive performance, but rather to understand and interpret how the model makes predictions.

The notebook includes:

- Data preprocessing
- Grid search model tuning
- Classification modeling
- Permutation importance analysis
- Partial dependence plots (PDPs)
- Interpretation of model behavior

---

## Model Workflow

### 1. Data Preprocessing

The dataset was cleaned and prepared by:

- Removing identifier columns
- Handling missing values
- Encoding categorical variables using `pd.get_dummies()`
- Converting the target variable (`Loan_Status`) into binary values

### 2. Train/Test Split

The data was divided into training and testing sets using:

- `train_test_split()`

with stratified sampling to preserve class balance.

### 3. Classification Model + Grid Search

A `RandomForestClassifier` was trained using `GridSearchCV`.

The following hyperparameters were tuned:

- `n_estimators`
- `max_depth`
- `min_samples_split`

This satisfies the assignment requirement of performing a grid search / automated model tuning process.

### 4. Permutation Importance

Permutation importance was used to identify the most influential features in the model.

The analysis used:

- `n_repeats = 20`

The **Top 5 most important features** were visualized using a box plot.

**Interpretation:**
- Larger decreases in accuracy indicate features that the model relies on more heavily.

### 5. Partial Dependence Plots (PDPs)

Partial dependence plots were created for the Top 5 features.

These plots help visualize how changes in a feature influence the model's predicted response while averaging over the other predictors.

For consistency:
- All PDP plots use the same Y-axis scale.

---

## Key Interpretation Insights

- `Credit_History` was the most influential feature in the model.
- Positive credit history strongly increased predicted loan approval probability.
- Some categorical variables had smaller but still measurable effects.
- Income-related variables were less influential than expected.
- Correlated predictors may affect permutation importance interpretation.

---

## Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

---

## Author

Yeongeun Ra
