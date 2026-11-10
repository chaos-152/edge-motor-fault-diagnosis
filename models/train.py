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
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.dataset_loader import StatorCurrentDataset
from dsp.feature_extractor import MCSAFeatureExtractor


META_COLS = ["label_binary", "label_multiclass", "load_torque_nm", "repetition", "recording_id"]


def export_tree_to_json(dt_clf: DecisionTreeClassifier, feature_names: List[str], out_path: Path) -> None:
    """Exports a fitted scikit-learn DecisionTreeClassifier to a dependency-free JSON structure."""
    tree = dt_clf.tree_
    payload = {
        "feature_names": list(feature_names),
        "classes": [int(c) for c in dt_clf.classes_],
        "children_left": [int(x) for x in tree.children_left],
        "children_right": [int(x) for x in tree.children_right],
        "feature": [int(x) for x in tree.feature],
        "threshold": [float(x) for x in tree.threshold],
        "value": [[[float(v) for v in row] for row in node] for node in tree.value],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)


def build_recording_ids(metadata: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts load_torques, repetitions, and unique recording_ids from sample metadata."""
    torques = np.array([float(m["load_torque_nm"]) for m in metadata], dtype=np.float32)
    reps = np.array([int(m.get("repetition", idx)) for idx, m in enumerate(metadata)], dtype=np.int64)
    rec_ids = np.array(
        [
            f"{int(m['fault_severity'])}_{float(m['load_torque_nm'])}_{int(m.get('repetition', idx))}"
            for idx, m in enumerate(metadata)
        ]
    )
    return torques, reps, rec_ids


class ModelTrainer:
    """
    Manages dataset feature extraction, leak-free grouped training, LOTO validation, and serialization.
    """

    def __init__(self, models_dir: Path = config.MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = MCSAFeatureExtractor()

    def prepare_data(
        self,
        samples_per_condition: int = 30,
        force_regenerate: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Loads cached features (if schema matches) or extracts MCSA features from cached/raw dataset.

        Returns:
            X: Feature matrix of shape (N, num_features)
            y_bin: Binary labels (0 = Healthy, 1 = Faulty)
            y_multi: Multiclass labels (0 to 4 Broken Bars)
            df_features: Pandas DataFrame with named features, labels, and recording metadata
        """
        dataset = StatorCurrentDataset()
        dataset_file = config.PROCESSED_DATA_DIR / "stator_current_dataset.npz"
        features_file = config.PROCESSED_DATA_DIR / "mcsa_features.csv"

        expected_cols = list(MCSAFeatureExtractor.FEATURE_NAMES)
        if not force_regenerate and features_file.exists():
            df = pd.read_csv(features_file)
            has_cols = all(col in df.columns for col in expected_cols + META_COLS)
            has_active_only = has_cols and float(df["load_current_pu"].min() * config.MOTOR_RATED_CURRENT_A) >= config.MIN_ACTIVE_CURRENT_A
            if has_cols and has_active_only:
                print(f"Loading cached MCSA features from {features_file}...")
                X = df[expected_cols].values.astype(np.float32)
                y_bin = df["label_binary"].values.astype(np.int64)
                y_multi = df["label_multiclass"].values.astype(np.int64)
                return X, y_bin, y_multi, df
            print("Feature schema updated or inactive rows detected; refreshing MCSA feature CSV...")

        if not force_regenerate and dataset_file.exists():
            print(f"Loading cached waveforms from {dataset_file}...")
            signals, labels_multi, labels_bin, metadata = dataset.load_processed_dataset()
            rms_all = np.sqrt(np.mean(signals ** 2, axis=1))
            active_mask = rms_all >= config.MIN_ACTIVE_CURRENT_A
            if not np.all(active_mask):
                dropped = int(np.sum(~active_mask))
                print(f"Filtered {dropped} inactive/unpowered windows (RMS < {config.MIN_ACTIVE_CURRENT_A} A).")
                signals = signals[active_mask]
                labels_multi = labels_multi[active_mask]
                labels_bin = labels_bin[active_mask]
                metadata = [m for m, keep in zip(metadata, active_mask) if keep]
                dataset.save_dataset(signals, labels_multi, labels_bin, metadata)
        elif (config.RAW_DATA_DIR / "struct_rs_R1.mat").exists():
            print("Detected official IEEE DataPort dataset in data/raw/! Processing empirical signals...")
            signals, labels_multi, labels_bin, metadata = dataset.load_ieee_dataport_dataset(
                windows_per_repetition=5
            )
            dataset.save_dataset(signals, labels_multi, labels_bin, metadata)
        else:
            print("Generating synthetic induction motor stator current signals...")
            signals, labels_multi, labels_bin, metadata = dataset.generate_benchmark_dataset(
                samples_per_condition=samples_per_condition
            )
            dataset.save_dataset(signals, labels_multi, labels_bin, metadata)

        print(f"Extracting MCSA spectral & time-domain features for {len(signals)} samples...")
        slips = [m.get("slip") for m in metadata]
        X = self.feature_extractor.extract_batch(signals, slips=slips)
        y_bin = labels_bin
        y_multi = labels_multi
        torques, reps, rec_ids = build_recording_ids(metadata)

        df = pd.DataFrame(X, columns=expected_cols)
        df["label_binary"] = y_bin
        df["label_multiclass"] = y_multi
        df["load_torque_nm"] = torques
        df["repetition"] = reps
        df["recording_id"] = rec_ids
        df.to_csv(features_file, index=False)
        print(f"Features successfully saved to {features_file}")

        return X, y_bin, y_multi, df

    @staticmethod
    def _split_by_recording(
        X: np.ndarray,
        y: np.ndarray,
        df_meta: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """Splits train/test strictly by independent physical recording to eliminate within-recording leakage."""
        if df_meta is not None and "repetition" in df_meta.columns and "recording_id" in df_meta.columns:
            reps = df_meta["repetition"].values
            rec_ids = df_meta["recording_id"].values
            holdout_mask = np.isin(reps, config.HOLDOUT_REPETITIONS)
            if np.any(holdout_mask) and np.any(~holdout_mask):
                return (
                    X[~holdout_mask], X[holdout_mask],
                    y[~holdout_mask], y[holdout_mask],
                    rec_ids[~holdout_mask], rec_ids[holdout_mask],
                    f"independent_repetitions_{config.HOLDOUT_REPETITIONS}",
                )
            gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SPLIT_RATIO, random_state=config.RANDOM_STATE)
            tr_idx, te_idx = next(gss.split(X, y, rec_ids))
            return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx], rec_ids[tr_idx], rec_ids[te_idx], "group_shuffle_split"

        # Fallback if metadata absent: index-unique groups
        rec_ids = np.arange(len(X))
        gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SPLIT_RATIO, random_state=config.RANDOM_STATE)
        tr_idx, te_idx = next(gss.split(X, y, rec_ids))
        return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx], rec_ids[tr_idx], rec_ids[te_idx], "sample_index_split"

    @staticmethod
    def _evaluate_loto(
        X: np.ndarray,
        y: np.ndarray,
        torques: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Runs Leave-One-Torque-Out evaluation across distinct operating load points."""
        loto = {"SVM": {}, "DecisionTree": {}}
        unique_torques = sorted(np.unique(torques))
        if len(unique_torques) < 2:
            return loto

        for t_val in unique_torques:
            mask_te = torques == t_val
            svm = Pipeline([("scaler", StandardScaler()), ("classifier", SVC(**config.SVM_CONFIG))])
            dt = DecisionTreeClassifier(**config.DECISION_TREE_CONFIG)
            svm.fit(X[~mask_te], y[~mask_te])
            dt.fit(X[~mask_te], y[~mask_te])
            loto["SVM"][f"{float(t_val):.1f}_Nm"] = float(accuracy_score(y[mask_te], svm.predict(X[mask_te])))
            loto["DecisionTree"][f"{float(t_val):.1f}_Nm"] = float(accuracy_score(y[mask_te], dt.predict(X[mask_te])))

        loto["SVM_mean"] = float(np.mean(list(loto["SVM"].values())))
        loto["DecisionTree_mean"] = float(np.mean(list(loto["DecisionTree"].values())))
        return loto

    def train_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_type: str = "binary",
        df_meta: Optional[pd.DataFrame] = None
    ) -> Dict[str, Dict]:
        """
        Trains SVM and Decision Tree classifiers with leak-free recording-grouped splits,
        grouped cross-validation, and Leave-One-Torque-Out (LOTO) generalization checks.
        """
        print(f"\n=======================================================")
        print(f"   TRAINING EDGE CLASSIFIERS — ZERO-LEAKAGE ({target_type.upper()})")
        print(f"=======================================================")

        X_train, X_test, y_train, y_test, g_train, g_test, split_mode = self._split_by_recording(X, y, df_meta)
        overlap = set(map(str, g_train)).intersection(set(map(str, g_test)))
        assert len(overlap) == 0, f"Recording leakage detected across split: {overlap}"

        print(f"Split strategy: {split_mode} (0 overlapping recordings between train & test)")
        print(f"Dataset split: {len(X_train)} training windows ({len(np.unique(g_train))} recordings), "
              f"{len(X_test)} unseen test windows ({len(np.unique(g_test))} recordings).")

        n_splits = min(config.CV_FOLDS, len(np.unique(g_train)))
        group_cv = GroupKFold(n_splits=n_splits)
        results = {"split_strategy": split_mode, "num_overlapping_recordings": 0}

        # 1. Support Vector Machine (SVM) Pipeline
        print("\n--- Training Support Vector Machine (SVM) ---")
        svm_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(**config.SVM_CONFIG))
        ])

        cv_scores_svm = cross_val_score(svm_pipe, X_train, y_train, groups=g_train, cv=group_cv, scoring="accuracy")
        svm_pipe.fit(X_train, y_train)
        y_pred_svm = svm_pipe.predict(X_test)
        acc_svm = accuracy_score(y_test, y_pred_svm)
        p_svm, r_svm, f1_svm, _ = precision_recall_fscore_support(y_test, y_pred_svm, average="weighted", zero_division=0)

        print(f"SVM {n_splits}-Fold GroupCV Accuracy: {cv_scores_svm.mean()*100:.2f}% (+/- {cv_scores_svm.std()*100:.2f}%)")
        print(f"SVM Holdout Test Accuracy:   {acc_svm*100:.2f}%")
        print(f"SVM Weighted F1-Score:       {f1_svm:.4f}")

        # 2. Decision Tree Classifier
        print("\n--- Training Decision Tree Classifier ---")
        dt_clf = DecisionTreeClassifier(**config.DECISION_TREE_CONFIG)

        cv_scores_dt = cross_val_score(dt_clf, X_train, y_train, groups=g_train, cv=group_cv, scoring="accuracy")
        dt_clf.fit(X_train, y_train)
        y_pred_dt = dt_clf.predict(X_test)
        acc_dt = accuracy_score(y_test, y_pred_dt)
        p_dt, r_dt, f1_dt, _ = precision_recall_fscore_support(y_test, y_pred_dt, average="weighted", zero_division=0)

        print(f"DT  {n_splits}-Fold GroupCV Accuracy: {cv_scores_dt.mean()*100:.2f}% (+/- {cv_scores_dt.std()*100:.2f}%)")
        print(f"DT  Holdout Test Accuracy:   {acc_dt*100:.2f}%")
        print(f"DT  Weighted F1-Score:       {f1_dt:.4f}")

        # Leave-One-Torque-Out Generalization
        loto_results = {}
        if df_meta is not None and "load_torque_nm" in df_meta.columns:
            loto_results = self._evaluate_loto(X, y, df_meta["load_torque_nm"].values)
            if loto_results:
                print(
                    f"Leave-One-Torque-Out Mean:   DT {loto_results['DecisionTree_mean']*100:.2f}% | "
                    f"SVM {loto_results['SVM_mean']*100:.2f}%"
                )

        # Inspect Decision Tree Rules for Edge Explainability
        tree_rules = export_text(dt_clf, feature_names=MCSAFeatureExtractor.FEATURE_NAMES)
        with open(self.models_dir / f"decision_tree_rules_{target_type}.txt", "w") as f_rules:
            f_rules.write(tree_rules)

        # Serialize Models for Edge Deployment (.joblib + dependency-free JSON for portable ARM64)
        suffix = f"_{target_type}" if target_type != "binary" else ""
        svm_path = self.models_dir / f"svm_model{suffix}.joblib"
        dt_path = self.models_dir / f"decision_tree_model{suffix}.joblib"
        dt_json_path = self.models_dir / f"decision_tree_portable{suffix}.json"

        joblib.dump(svm_pipe, svm_path)
        joblib.dump(dt_clf, dt_path)
        export_tree_to_json(dt_clf, MCSAFeatureExtractor.FEATURE_NAMES, dt_json_path)

        print(f"\nModels exported to:")
        print(f"  -> {svm_path} ({svm_path.stat().st_size / 1024:.1f} KB)")
        print(f"  -> {dt_path} ({dt_path.stat().st_size / 1024:.1f} KB)")
        print(f"  -> {dt_json_path} ({dt_json_path.stat().st_size / 1024:.1f} KB)")

        results["SVM"] = {
            "cv_accuracy_mean": float(cv_scores_svm.mean()),
            "cv_accuracy_std": float(cv_scores_svm.std()),
            "test_accuracy": float(acc_svm),
            "test_precision": float(p_svm),
            "test_recall": float(r_svm),
            "test_f1": float(f1_svm),
            "loto_accuracy_mean": loto_results.get("SVM_mean"),
            "loto_per_torque": loto_results.get("SVM", {}),
            "confusion_matrix": confusion_matrix(y_test, y_pred_svm).tolist(),
            "model_path": str(svm_path),
            "model_size_kb": float(svm_path.stat().st_size / 1024.0),
            "y_test": y_test.tolist(),
            "y_pred": y_pred_svm.tolist(),
        }

        results["DecisionTree"] = {
            "cv_accuracy_mean": float(cv_scores_dt.mean()),
            "cv_accuracy_std": float(cv_scores_dt.std()),
            "test_accuracy": float(acc_dt),
            "test_precision": float(p_dt),
            "test_recall": float(r_dt),
            "test_f1": float(f1_dt),
            "loto_accuracy_mean": loto_results.get("DecisionTree_mean"),
            "loto_per_torque": loto_results.get("DecisionTree", {}),
            "confusion_matrix": confusion_matrix(y_test, y_pred_dt).tolist(),
            "model_path": str(dt_path),
            "model_size_kb": float(dt_path.stat().st_size / 1024.0),
            "y_test": y_test.tolist(),
            "y_pred": y_pred_dt.tolist(),
        }

        summary_path = config.REPORTS_DIR / f"training_metrics_{target_type}.json"
        with open(summary_path, "w") as f_json:
            json.dump(results, f_json, indent=2)

        return results


if __name__ == "__main__":
    trainer = ModelTrainer()
    X, y_bin, y_multi, df = trainer.prepare_data(samples_per_condition=30)
    results_bin = trainer.train_models(X, y_bin, target_type="binary", df_meta=df)
    results_multi = trainer.train_models(X, y_multi, target_type="multiclass", df_meta=df)
