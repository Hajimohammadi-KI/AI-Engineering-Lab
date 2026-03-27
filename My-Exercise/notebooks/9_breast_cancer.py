import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the breast cancer dataset
breast_cancer = load_breast_cancer()
df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

# Add target column
df['target'] = breast_cancer.target

# Filter to keep only 'worst' features and target
df = df[[col for col in df.columns if 'worst' in col] + ['target']]

# Clean column names by removing 'worst ' prefix
df.columns = df.columns.str.replace('worst ', '')

# Display target distribution
vs = df['target'].value_counts()
vs.plot(kind='bar', figsize=(3, 3))
plt.show()

# Visualize radius vs target
plt.figure(figsize=(4, 4))
sns.boxplot(x='target', y='radius', data=df)
plt.show()

# Visualize concave points vs target
plt.figure(figsize=(4, 4))
sns.boxplot(x='target', y='concave points', data=df)
plt.show()

# Correlation analysis
print(df.corr())

# Correlation heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(df.corr().abs(), annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

# Drop redundant features (perimeter and area are highly correlated with radius)
df = df.drop(columns=['perimeter', 'area'])

# Data quality checks
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nDescriptive statistics:\n{df.describe()}")

# Box plot for concave points
plt.figure(figsize=(4, 4))
sns.boxplot(x='target', y='concave points', data=df)
plt.show()

# Box plot for radius
sns.boxplot(data=df, y='radius')
plt.show()

# Radius statistics
print(f"\nRadius statistics:\n{df['radius'].describe()}")

# Pair plot with target hue
sns.pairplot(df, hue='target')
plt.show()

# Feature selection based on correlation with target
print(f"\nCorrelation with target:\n{df.corr()}")
c = df.corr()['target'].drop('target')
fs = c.abs().sort_values(ascending=False).index.tolist()
print(f"Features ranked by correlation: {fs}")

# Data preparation
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Total: {X_train.shape[0] + X_test.shape[0]}")

# Model training
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Confusion matrix
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Total correct predictions: {39 + 70} (from confusion matrix)")

# Model evaluation metrics
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
