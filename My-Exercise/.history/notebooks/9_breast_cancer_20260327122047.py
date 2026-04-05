"""Exploratory analysis and baseline model for breast cancer classification."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_prepared_dataframe() -> pd.DataFrame:
	"""Load dataset and keep only 'worst' features plus target label."""
	dataset = load_breast_cancer()
	frame = pd.DataFrame(dataset.data, columns=dataset.feature_names)
	frame["target"] = dataset.target

	frame = frame[[col for col in frame.columns if "worst" in col] + ["target"]]
	frame.columns = frame.columns.str.replace("worst ", "", regex=False)
	return frame


def run_eda(df: pd.DataFrame) -> pd.DataFrame:
	"""Run visual and statistical checks and return cleaned dataframe."""
	sns.set_theme(style="whitegrid")

	df["target"].value_counts().plot(kind="bar", figsize=(4, 4), title="Target Distribution")
	plt.tight_layout()
	plt.show()

	plt.figure(figsize=(5, 4))
	sns.boxplot(x="target", y="radius", data=df)
	plt.title("Radius by Target")
	plt.tight_layout()
	plt.show()

	plt.figure(figsize=(5, 4))
	sns.boxplot(x="target", y="concave points", data=df)
	plt.title("Concave Points by Target")
	plt.tight_layout()
	plt.show()

	corr_matrix = df.corr(numeric_only=True)
	print("Correlation matrix:\n", corr_matrix)

	plt.figure(figsize=(9, 7))
	sns.heatmap(corr_matrix.abs(), annot=True, fmt=".2f", cmap="coolwarm")
	plt.title("Absolute Correlation Heatmap")
	plt.tight_layout()
	plt.show()

	# Perimeter and area are strongly correlated with radius and can be dropped.
	cleaned_df = df.drop(columns=["perimeter", "area"])

	print("\nMissing values:\n", cleaned_df.isnull().sum())
	print("\nDescriptive statistics:\n", cleaned_df.describe())
	print("\nRadius statistics:\n", cleaned_df["radius"].describe())

	return cleaned_df


def train_and_evaluate(df: pd.DataFrame) -> None:
	"""Train logistic regression baseline and print evaluation metrics."""
	X = df.drop(columns=["target"])
	y = df["target"]

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=TEST_SIZE,
		random_state=RANDOM_STATE,
		stratify=y,
	)

	print(f"Training set size: {X_train.shape}")
	print(f"Test set size: {X_test.shape}")
	print(f"Total samples: {X_train.shape[0] + X_test.shape[0]}")

	model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	cm = confusion_matrix(y_test, y_pred)
	print("\nConfusion Matrix:\n", cm)
	print(f"Total correct predictions: {int(cm[0, 0] + cm[1, 1])}")

	print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
	print(f"Precision: {precision_score(y_test, y_pred):.4f}")
	print(f"Recall: {recall_score(y_test, y_pred):.4f}")
	print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")


def main() -> None:
	"""Entry point for script execution."""
	df = load_prepared_dataframe()
	df = run_eda(df)

	corr_with_target = df.corr(numeric_only=True)["target"].drop("target")
	ranked_features = corr_with_target.abs().sort_values(ascending=False).index.tolist()
	print("\nFeatures ranked by absolute correlation with target:", ranked_features)

	train_and_evaluate(df)


if __name__ == "__main__":
	main()
