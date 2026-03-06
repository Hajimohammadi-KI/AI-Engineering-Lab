# ============================================
# K-Fold Cross-Validation and Hold-Out Validation
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("K-Fold Cross-Validation and Hold-Out Validation")
print("="*60)

# ============================================
# LOAD DATASET
# ============================================
iris = load_iris()
X = iris.data
y = iris.target

print(f"\nDataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")

# ============================================
# PART 1: HOLD-OUT VALIDATION
# ============================================
print("\n" + "="*60)
print("PART 1: HOLD-OUT VALIDATION")
print("="*60)

def hold_out_split(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    """Split data into training, validation, and test sets."""
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )
    # Second split: separate train and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# Apply hold-out split
X_train, X_val, X_test, y_train, y_val, y_test = hold_out_split(X, y)

print(f"\nHold-Out Split Results:")
print(f"  Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"  Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Train model with Hold-Out
model_holdout = LogisticRegression(max_iter=200)
model_holdout.fit(X_train, y_train)

train_accuracy = accuracy_score(y_train, model_holdout.predict(X_train))
val_accuracy = accuracy_score(y_val, model_holdout.predict(X_val))
y_test_pred = model_holdout.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nHold-Out Training Results:")
print(f"  1. Training Accuracy:   {train_accuracy:.4f}")
print(f"  2. Validation Accuracy: {val_accuracy:.4f}")
print(f"  3. Test Accuracy:       {test_accuracy:.4f}")

# ============================================
# PART 2: K-FOLD CROSS-VALIDATION
# ============================================
print("\n" + "="*60)
print("PART 2: K-FOLD CROSS-VALIDATION")
print("="*60)

def k_fold_cross_validation(X, y, model, k=5, random_state=42):
    """Perform K-Fold Cross-Validation manually."""
    kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_scores = []

    print(f"\nK-Fold Cross-Validation (K={k})")
    print("-" * 50)

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model_fold = clone(model)
        model_fold.fit(X_train_fold, y_train_fold)

        train_score = accuracy_score(y_train_fold, model_fold.predict(X_train_fold))
        val_score = accuracy_score(y_val_fold, model_fold.predict(X_val_fold))
        fold_scores.append(val_score)

        print(f"  Fold {fold_idx}: Train Acc = {train_score:.4f}, Val Acc = {val_score:.4f}")

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print("-" * 50)
    print(f"  Mean Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")

    return {'fold_scores': fold_scores, 'mean': mean_score, 'std': std_score}

# Run K-Fold
model_kfold = LogisticRegression(max_iter=200)
results = k_fold_cross_validation(X, y, model_kfold, k=5)

# Using sklearn's cross_val_score
print("\nUsing sklearn's cross_val_score:")
cv_scores = cross_val_score(LogisticRegression(max_iter=200), X, y, cv=5)
print(f"  Fold scores: {cv_scores}")
print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================
# PART 3: COMPARISON TABLE
# ============================================
print("\n" + "="*60)
print("COMPARISON: Hold-Out vs K-Fold Cross-Validation")
print("="*60)
print(f"\n{'Aspect':<25} {'Hold-Out':<25} {'K-Fold CV':<25}")
print("-"*70)
print(f"{'Data Usage':<25} {'Single split':<25} {'All data used K times':<25}")
print(f"{'Training Iterations':<25} {'1':<25} {'K':<25}")
print(f"{'Variance in Estimate':<25} {'Higher':<25} {'Lower':<25}")
print(f"{'Computational Cost':<25} {'Lower':<25} {'Higher (K times)':<25}")
print(f"{'Best For':<25} {'Large datasets':<25} {'Small/medium datasets':<25}")

# ============================================
# PART 4: MODEL COMPARISON
# ============================================
print("\n" + "="*60)
print("MODEL COMPARISON USING BOTH METHODS")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf')
}

print(f"\n{'Model':<25} {'Hold-Out (Test)':<20} {'5-Fold CV (Mean +/- Std)':<25}")
print("-"*70)

for name, model in models.items():
    model.fit(X_train, y_train)
    holdout_score = accuracy_score(y_test, model.predict(X_test))
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"{name:<25} {holdout_score:<20.4f} {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================
# PART 5: VISUALIZATION
# ============================================
print("\n" + "="*60)
print("Creating visualizations...")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Hold-Out Results
ax1 = axes[0]
sets = ['Training', 'Validation', 'Test']
holdout_scores = [train_accuracy, val_accuracy, test_accuracy]
colors = ['#2ecc71', '#3498db', '#e74c3c']
bars1 = ax1.bar(sets, holdout_scores, color=colors, edgecolor='black')
ax1.set_ylim(0, 1.1)
ax1.set_ylabel('Accuracy')
ax1.set_title('Hold-Out Validation Results')
for bar, score in zip(bars1, holdout_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{score:.3f}', ha='center')

# Plot 2: K-Fold Results
ax2 = axes[1]
folds = [f'Fold {i+1}' for i in range(5)]
bars2 = ax2.bar(folds, results['fold_scores'], color='#9b59b6', edgecolor='black')
ax2.axhline(y=results['mean'], color='red', linestyle='--', label=f"Mean = {results['mean']:.3f}")
ax2.set_ylim(0, 1.1)
ax2.set_ylabel('Accuracy')
ax2.set_title('5-Fold Cross-Validation Results')
ax2.legend()
for bar, score in zip(bars2, results['fold_scores']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{score:.3f}', ha='center')

plt.tight_layout()
plt.savefig('comparison_plot.png', dpi=150)
plt.show()

# K-Fold Visualization
from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(12, 4))
k = 5

for fold in range(k):
    for i in range(k):
        color = '#e74c3c' if i == fold else '#2ecc71'
        rect = plt.Rectangle((i, k-1-fold), 0.9, 0.8, color=color, edgecolor='black')
        ax.add_patch(rect)

ax.set_xlim(-0.5, k+1)
ax.set_ylim(-0.5, k+0.5)
ax.set_xlabel('Data Folds')
ax.set_ylabel('Iteration')
ax.set_title('K-Fold Cross-Validation Visualization (K=5)')
ax.set_xticks([i+0.45 for i in range(k)])
ax.set_xticklabels([f'Fold {i+1}' for i in range(k)])
ax.set_yticks([i+0.4 for i in range(k)])
ax.set_yticklabels([f'Iter {k-i}' for i in range(k)])

legend_elements = [Patch(facecolor='#2ecc71', edgecolor='black', label='Training'),
                   Patch(facecolor='#e74c3c', edgecolor='black', label='Validation')]
ax.legend(handles=legend_elements, loc='upper right')
plt.tight_layout()
plt.savefig('kfold_visualization.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("DONE! Plots saved as PNG files.")
print("="*60)
