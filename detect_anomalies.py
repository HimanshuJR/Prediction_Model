import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from src.helpers import load_data, load_model

def detect_anomalies(X, y_true, y_pred, threshold_factor=3):
    """
    Compute residuals and flag anomalies based on dynamic thresholds.
    """
    residuals = np.abs(y_true - y_pred)
    thresholds = residuals.mean(axis=0) + threshold_factor * residuals.std(axis=0)
    anomalies = (residuals > thresholds).astype(int)  # 1 = anomaly, 0 = normal
    return residuals, thresholds, anomalies

if __name__ == "__main__":
    # Load test data and model
    X_test, y_test = load_data(split='test')  # Ensure this returns numpy arrays or DataFrames
    model_container = load_model("models/trained_model.pkl")

    # Extract actual model if model_container is a dict
    if isinstance(model_container, dict):
        model = model_container.get('model')
        if model is None:
            raise ValueError("Key 'model' not found in loaded model dictionary.")
    else:
        model = model_container

    # Make predictions
    y_pred = model.predict(X_test)

    # Detect anomalies
    residuals, thresholds, anomaly_flags = detect_anomalies(X_test, y_test, y_pred)

    # Save results
    residuals_df = pd.DataFrame(residuals, columns=[f"ServoLoad_{i}_residual" for i in range(y_test.shape[1])])
    anomaly_df = pd.DataFrame(anomaly_flags, columns=[f"ServoLoad_{i}_anomaly" for i in range(y_test.shape[1])])
    output = pd.concat([residuals_df, anomaly_df], axis=1)
    output.to_csv("data/anomaly_flags.csv", index=False)

    print("✅ Anomaly detection complete. Results saved to data/anomaly_flags.csv")
