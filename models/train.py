"""
Machine Learning Training Pipeline for Induction Motor Fault Diagnosis.
Trains edge-viable classifiers using scikit-learn:
1. Support Vector Machine (SVM)
2. Decision Tree Classifier
Ref: "Edge vs. Cloud: Empirical Insights into Data-Driven Condition Monitoring" (Walani & Doorsamy, 2025)
"""

import sys
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.dataset_loader import StatorCurrentDataset
from dsp.feature_extractor import MCSAFeatureExtractor


class ModelTrainer:
    """
    Manages dataset feature extraction, training, validation, and serialization.
    """

    def __init__(self, models_dir: Path = config.MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = MCSAFeatureExtractor()

    def prepare_data(
        self,
        samples_per_condition: int = 40,
        force_regenerate: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Loads cached features or generates dataset and extracts MCSA features.

        Returns:
            X: Feature matrix of shape (N, num_features)
            y_bin: Binary labels (0 = Healthy, 1 = Faulty)
            y_multi: Multiclass labels (0 to 4 Broken Bars)
            df_features: Pandas DataFrame with named features and labels
        """
        dataset = StatorCurrentDataset()
        dataset_file = config.PROCESSED_DATA_DIR / "stator_current_dataset.npz"
        features_file = config.PROCESSED_DATA_DIR / "mcsa_features.csv"

        if not force_regenerate and features_file.exists():
            print(f"Loading cached MCSA features from {features_file}...")
            df = pd.read_csv(features_file)
            feature_cols = [c for c in df.columns if c not in ["label_binary", "label_multiclass", "load_torque_nm"]]
            X = df[feature_cols].values.astype(np.float32)
            y_bin = df["label_binary"].values.astype(np.int64)
            y_multi = df["label_multiclass"].values.astype(np.int64)
            return X, y_bin, y_multi, df

        has_ieee_files = (config.RAW_DATA_DIR / "struct_rs_R1.mat").exists()

        if has_ieee_files:
            print("Detected official IEEE DataPort dataset in data/raw/! Processing empirical signals...")
            signals, labels_multi, labels_bin, metadata = dataset.load_ieee_dataport_dataset(
                windows_per_repetition=samples_per_condition // 5 if samples_per_condition >= 10 else 2
            )
        else:
            print("Generating induction motor stator current signals...")
            signals, labels_multi, labels_bin, metadata = dataset.generate_benchmark_dataset(
                samples_per_condition=samples_per_condition
            )

        dataset.save_dataset(signals, labels_multi, labels_bin, metadata)

        print(f"Extracting MCSA spectral & time-domain features for {len(signals)} samples...")
        slips = [m.get("slip") for m in metadata]
        X = self.feature_extractor.extract_batch(signals, slips=slips)
        y_bin = labels_bin
        y_multi = labels_multi

        # Save to CSV for analysis and transparency
        df = pd.DataFrame(X, columns=MCSAFeatureExtractor.FEATURE_NAMES)
        df["label_binary"] = y_bin
        df["label_multiclass"] = y_multi
        df["load_torque_nm"] = [m["load_torque_nm"] for m in metadata]
        df.to_csv(features_file, index=False)
        print(f"Features successfully saved to {features_file}")

        return X, y_bin, y_multi, df

    def train_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_type: str = "binary"
    ) -> Dict[str, Dict]:
        """
        Trains SVM and Decision Tree classifiers with cross-validation.
        """
        print(f"\n=======================================================")
        print(f"   TRAINING EDGE-VIABLE CLASSIFIERS ({target_type.upper()})")
        print(f"=======================================================")

        # Train/Test Split (80% train, 20% test, stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SPLIT_RATIO, random_state=config.RANDOM_STATE, stratify=y
        )
        print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} testing samples.")

        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        results = {}

        # 1. Support Vector Machine (SVM) Pipeline
        print("\n--- Training Support Vector Machine (SVM) ---")
        svm_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(**config.SVM_CONFIG))
        ])

        cv_scores_svm = cross_val_score(svm_pipe, X_train, y_train, cv=cv, scoring="accuracy")
        svm_pipe.fit(X_train, y_train)
        y_pred_svm = svm_pipe.predict(X_test)
        acc_svm = accuracy_score(y_test, y_pred_svm)
        p_svm, r_svm, f1_svm, _ = precision_recall_fscore_support(y_test, y_pred_svm, average="weighted")

        print(f"SVM 5-Fold CV Accuracy: {cv_scores_svm.mean()*100:.2f}% (+/- {cv_scores_svm.std()*100:.2f}%)")
        print(f"SVM Test Accuracy:      {acc_svm*100:.2f}%")
        print(f"SVM Weighted F1-Score:  {f1_svm:.4f}")

        # 2. Decision Tree Classifier
        print("\n--- Training Decision Tree Classifier ---")
        dt_clf = DecisionTreeClassifier(**config.DECISION_TREE_CONFIG)

        cv_scores_dt = cross_val_score(dt_clf, X_train, y_train, cv=cv, scoring="accuracy")
        dt_clf.fit(X_train, y_train)
        y_pred_dt = dt_clf.predict(X_test)
        acc_dt = accuracy_score(y_test, y_pred_dt)
        p_dt, r_dt, f1_dt, _ = precision_recall_fscore_support(y_test, y_pred_dt, average="weighted")

        print(f"Decision Tree 5-Fold CV Accuracy: {cv_scores_dt.mean()*100:.2f}% (+/- {cv_scores_dt.std()*100:.2f}%)")
        print(f"Decision Tree Test Accuracy:      {acc_dt*100:.2f}%")
        print(f"Decision Tree Weighted F1-Score:  {f1_dt:.4f}")

        # Inspect Decision Tree Rules for Edge Explainability
        tree_rules = export_text(dt_clf, feature_names=MCSAFeatureExtractor.FEATURE_NAMES)
        with open(self.models_dir / f"decision_tree_rules_{target_type}.txt", "w") as f_rules:
            f_rules.write(tree_rules)

        # Serialize Models for Raspberry Pi Edge Node
        suffix = f"_{target_type}" if target_type != "binary" else ""
        svm_path = self.models_dir / f"svm_model{suffix}.joblib"
        dt_path = self.models_dir / f"decision_tree_model{suffix}.joblib"

        joblib.dump(svm_pipe, svm_path)
        joblib.dump(dt_clf, dt_path)
        print(f"\nModels exported to:")
        print(f"  -> {svm_path} ({svm_path.stat().st_size / 1024:.1f} KB)")
        print(f"  -> {dt_path} ({dt_path.stat().st_size / 1024:.1f} KB)")

        results["SVM"] = {
            "cv_accuracy_mean": float(cv_scores_svm.mean()),
            "cv_accuracy_std": float(cv_scores_svm.std()),
            "test_accuracy": float(acc_svm),
            "test_precision": float(p_svm),
            "test_recall": float(r_svm),
            "test_f1": float(f1_svm),
            "confusion_matrix": confusion_matrix(y_test, y_pred_svm).tolist(),
            "model_path": str(svm_path),
            "y_test": y_test.tolist(),
            "y_pred": y_pred_svm.tolist()
        }

        results["DecisionTree"] = {
            "cv_accuracy_mean": float(cv_scores_dt.mean()),
            "cv_accuracy_std": float(cv_scores_dt.std()),
            "test_accuracy": float(acc_dt),
            "test_precision": float(p_dt),
            "test_recall": float(r_dt),
            "test_f1": float(f1_dt),
            "confusion_matrix": confusion_matrix(y_test, y_pred_dt).tolist(),
            "model_path": str(dt_path),
            "y_test": y_test.tolist(),
            "y_pred": y_pred_dt.tolist()
        }

        # Save metrics summary
        summary_path = config.REPORTS_DIR / f"training_metrics_{target_type}.json"
        with open(summary_path, "w") as f_json:
            json.dump(results, f_json, indent=2)

        return results


if __name__ == "__main__":
    trainer = ModelTrainer()
    X, y_bin, y_multi, df = trainer.prepare_data(samples_per_condition=30)
    # Train primary binary model (Healthy vs. Faulty) as highlighted in Slide 4
    results_bin = trainer.train_models(X, y_bin, target_type="binary")
    # Also train multiclass model (Severity 0 to 4)
    results_multi = trainer.train_models(X, y_multi, target_type="multiclass")
