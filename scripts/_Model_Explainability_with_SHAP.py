# Cell 1: Install SHAP (if not installed)
# !pip install shap

# Cell 2: Imports
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier

# Cell 3: Load test data and trained model
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Load the trained Random Forest model (you may need to save it in model_training.py using joblib)
rf_model = joblib.load("rf_model.pkl")

# Cell 4: Create SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)[1]  # class 1 = fraud

# Cell 5: Global feature importance (summary plot)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

# Cell 6: Detailed global feature influence (beeswarm plot)
shap.summary_plot(shap_values, X_test, show=True)

# Cell 7: Local explanation for one prediction (waterfall plot)
sample_index = 0
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[sample_index],
        base_values=explainer.expected_value[1],
        data=X_test.iloc[sample_index],
        feature_names=X_test.columns.tolist()
    )
)

