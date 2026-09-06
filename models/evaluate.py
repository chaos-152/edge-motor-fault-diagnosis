"""Model evaluation and metrics reporting."""
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def evaluate_model_pipeline(model_path: Path, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Evaluates serialized model on test data and returns detailed metrics."""
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(report_dict["accuracy"]),
        "weighted_avg": report_dict["weighted avg"],
        "confusion_matrix": cm.tolist(),
        "classes": np.unique(y_test).tolist()
    }

    if y_prob is not None and len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))

    return metrics
