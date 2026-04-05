"""Training script for breast cancer classification.

Run:
	python -m src.train
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_project_root() -> Path:
	"""Return project root based on this file location."""
	return Path(__file__).resolve().parent.parent


def load_data() -> tuple[Any, Any, list[str]]:
	"""Load features, labels, and feature names from sklearn dataset."""
	dataset = load_breast_cancer()
	X = dataset.data
	y = dataset.target
	feature_names = list(dataset.feature_names)
	return X, y, feature_names


def build_model() -> Pipeline:
	"""Create a pipeline with scaling and logistic regression."""
	return Pipeline(
		steps=[
			("scaler", StandardScaler()),
			(
				"model",
				LogisticRegression(
					max_iter=1000,
					random_state=RANDOM_STATE,
					solver="lbfgs",
				),
			),
		]
	)


def evaluate_model(model: Pipeline, X_test: Any, y_test: Any) -> dict[str, float]:
	"""Compute standard classification metrics on the test split."""
	y_pred = model.predict(X_test)
	return {
		"accuracy": float(accuracy_score(y_test, y_pred)),
		"precision": float(precision_score(y_test, y_pred)),
		"recall": float(recall_score(y_test, y_pred)),
		"f1_score": float(f1_score(y_test, y_pred)),
	}


def save_artifacts(model: Pipeline, metrics: dict[str, float], feature_names: list[str]) -> None:
	"""Persist trained model and metrics for downstream prediction."""
	root = get_project_root()
	models_dir = root / "models"
	outputs_dir = root / "outputs"
	models_dir.mkdir(parents=True, exist_ok=True)
	outputs_dir.mkdir(parents=True, exist_ok=True)

	model_path = models_dir / "breast_cancer_logreg.joblib"
	metadata_path = models_dir / "model_metadata.json"
	metrics_path = outputs_dir / "metrics.json"

	joblib.dump(model, model_path)

	metadata = {
		"model_type": "LogisticRegression",
		"features": feature_names,
		"random_state": RANDOM_STATE,
		"test_size": TEST_SIZE,
	}

	metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
	metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
	"""Train model, evaluate it, and save artifacts."""
	X, y, feature_names = load_data()

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=TEST_SIZE,
		random_state=RANDOM_STATE,
		stratify=y,
	)

	model = build_model()
	model.fit(X_train, y_train)

	metrics = evaluate_model(model, X_test, y_test)
	save_artifacts(model, metrics, feature_names)

	print("Training completed.")
	for metric_name, value in metrics.items():
		print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
	main()
