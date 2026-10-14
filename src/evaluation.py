"""Model evaluation metrics and bias-variance decomposition."""

import numpy as np


def evaluate_regression(y_true, y_pred):
    """Compute MSE, RMSE, and R2 performance metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - sse / sst) if sst > 1e-12 else 0.0
    return {'MSE': mse, 'RMSE': rmse, 'R2': r2}


def compute_directional_accuracy(y_true, y_pred):
    """Evaluate binary directional hit rate (sign prediction)."""
    actual_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    return float(np.mean(actual_dir == pred_dir))
