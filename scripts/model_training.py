"""
Script: model_training.py
Purpose: Train and evaluate Logistic Regression and Random Forest models on processed fraud datasets.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, classification_report
from sklearn.model_selection import train_test_split


# Load processed data (numeric features only, to match SMOTE output)
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_train = pd.read_csv('data/processed/y_train_res.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# Logistic Regression
print("\n--- Logistic Regression ---")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

print("F1-Score:", f1_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("AUC-PR:", average_precision_score(y_test, y_proba_lr))
print(classification_report(y_test, y_pred_lr))

# Random Forest
print("\n--- Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("F1-Score:", f1_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("AUC-PR:", average_precision_score(y_test, y_proba_rf))
print(classification_report(y_test, y_pred_rf))

# Model selection justification
print("\n--- Model Selection Justification ---")
print("Compare F1-Score and AUC-PR. Random Forest is expected to perform better on imbalanced data due to its ensemble nature, but Logistic Regression offers interpretability. Choose the model with the best balance of performance and interpretability for your use case.")

import joblib

# Save the trained Random Forest model
joblib.dump(rf, 'rf_model.pkl')
print("Random Forest model saved as rf_model.pkl")