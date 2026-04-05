"""Prediction script for breast cancer classification.

Run sample predictions:
	python -m src.predict --samples 5

Run predictions from CSV:
	python -m src.predict --input path/to/data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_project_root() -> Path:
	"""Return project root based on this file location."""
	return Path(__file__).resolve().parent.parent


def load_model(model_path: Path | None = None):
	"""Load serialized model from models directory."""
	resolved_path = model_path or (get_project_root() / "models" / "breast_cancer_logreg.joblib")
	if not resolved_path.exists():
		raise FileNotFoundError(
			f"Model not found at {resolved_path}. Run 'python -m src.train' first."
		)
	return joblib.load(resolved_path)


def load_reference_test_data() -> tuple[pd.DataFrame, pd.Series]:
	"""Load the dataset and return the held-out test split for demo predictions."""
	dataset = load_breast_cancer()
	X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
	y = pd.Series(dataset.target, name="target")
	_, X_test, _, y_test = train_test_split(
		X,
		y,
		test_size=TEST_SIZE,
		random_state=RANDOM_STATE,
		stratify=y,
	)
	return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def predict_from_dataframe(model, features: pd.DataFrame) -> pd.DataFrame:
	"""Return predictions and probabilities for a feature dataframe."""
	predicted_class = model.predict(features)
	predicted_probability = model.predict_proba(features)[:, 1]

	result = features.copy()
	result["predicted_class"] = predicted_class
	result["predicted_probability"] = predicted_probability
	return result


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for prediction mode."""
	parser = argparse.ArgumentParser(description="Run breast cancer model predictions.")
	parser.add_argument(
		"--input",
		type=str,
		default=None,
		help="Optional CSV file with feature columns matching training data.",
	)
	parser.add_argument(
		"--samples",
		type=int,
		default=5,
		help="Number of demo samples from the test split when no input CSV is provided.",
	)
	return parser.parse_args()


def main() -> None:
	"""Run predictions from either CSV input or built-in test samples."""
	args = parse_args()
	model = load_model()

	if args.input:
		input_path = Path(args.input)
		if not input_path.exists():
			raise FileNotFoundError(f"Input CSV not found: {input_path}")

		features = pd.read_csv(input_path)
		predictions = predict_from_dataframe(model, features)
		print(predictions.head(10).to_string(index=False))
		return

	X_test, y_test = load_reference_test_data()
	n_samples = max(1, min(args.samples, len(X_test)))
	sample_features = X_test.head(n_samples)
	sample_labels = y_test.head(n_samples)

	predictions = predict_from_dataframe(model, sample_features)
	predictions["actual_class"] = sample_labels.values

	print(predictions.to_string(index=False))


if __name__ == "__main__":
	main()
