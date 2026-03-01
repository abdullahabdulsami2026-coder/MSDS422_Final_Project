"""
03b_random_forest.py
=====================
Random Forest Model for Diabetes 30-Day Readmission Prediction

Ensemble model with built-in feature importance and class balancing.

Author: Abdullah Abdulsami
Team: Final Project - G4 (Abdullah, Jiahao, Feifan)
Course: MSDS 422 - Practical Machine Learning
Date: February 2026
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
)
import joblib

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

# ============================================================
# Configuration
# ============================================================
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("outputs", "random_forest")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RANDOM_STATE = 42
N_FOLDS = 5


def load_processed_data():
    """Load preprocessed data (unscaled — RF doesn't need scaling)."""
    print("Loading preprocessed data (unscaled)...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_unscaled.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_unscaled.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]:,} samples")
    print(f"  Positive rate (train): {y_train.mean():.3f}")
    return X_train, X_test, y_train, y_test


def tune_and_train(X_train, y_train):
    """Hyperparameter tuning for Random Forest."""
    print("\n--- Hyperparameter Tuning (GridSearchCV) ---")

    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Use a smaller grid for speed — randomized search with key combos
    from sklearn.model_selection import RandomizedSearchCV

    grid = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid, n_iter=30, cv=cv, scoring="roc_auc",
        n_jobs=-1, verbose=1, refit=True, random_state=RANDOM_STATE,
    )
    grid.fit(X_train, y_train)

    print(f"\n  Best AUC-ROC (CV): {grid.best_score_:.4f}")
    print(f"  Best parameters: {grid.best_params_}")
    return grid.best_estimator_, grid.best_params_


def cross_validate_model(model, X_train, y_train):
    """5-fold cross-validation metrics."""
    print(f"\n--- {N_FOLDS}-Fold Cross-Validation ---")
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    y_proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "cv_auc_roc": roc_auc_score(y_train, y_proba),
        "cv_accuracy": accuracy_score(y_train, y_pred),
        "cv_precision": precision_score(y_train, y_pred),
        "cv_recall": recall_score(y_train, y_pred),
        "cv_f1": f1_score(y_train, y_pred),
        "cv_avg_precision": average_precision_score(y_train, y_proba),
    }

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def evaluate_on_test(model, X_test, y_test):
    """Evaluate on hold-out test set."""
    print("\n--- Test Set Evaluation ---")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "test_auc_roc": roc_auc_score(y_test, y_proba),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_avg_precision": average_precision_score(y_test, y_proba),
    }

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["No Readmit", "Readmit <30d"]))

    return metrics, y_pred, y_proba


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot Random Forest feature importance (Gini impurity)."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
        "Std": std,
    }).sort_values("Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(len(feat_imp)), feat_imp["Importance"].values,
        xerr=feat_imp["Std"].values,
        color=sns.color_palette("YlOrRd_r", len(feat_imp)),
        capsize=3,
    )
    ax.set_yticks(range(len(feat_imp)))
    ax.set_yticklabels(feat_imp["Feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("Random Forest Feature Importance — Top 20 (with Std Dev)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "rf_feature_importance.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Save CSV
    feat_imp_full = pd.DataFrame({
        "Feature": feature_names, "Importance": importances,
    }).sort_values("Importance", ascending=False)
    feat_imp_full.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)


def plot_roc_curve(y_test, y_proba):
    """Plot ROC curve."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#f59e0b", linewidth=2, label=f"Random Forest (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#f59e0b")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Random Forest")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "roc_curve.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="YlOrBr",
                xticklabels=["No Readmit", "Readmit <30d"],
                yticklabels=["No Readmit", "Readmit <30d"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Random Forest")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "confusion_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("RANDOM FOREST MODEL PIPELINE")
    print("Diabetes 30-Day Readmission Prediction")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_processed_data()
    feature_names = X_train.columns.tolist()

    best_model, best_params = tune_and_train(X_train, y_train)
    cv_metrics = cross_validate_model(best_model, X_train, y_train)
    test_metrics, y_pred, y_proba = evaluate_on_test(best_model, X_test, y_test)

    # Plots
    plot_feature_importance(best_model, feature_names)
    plot_roc_curve(y_test, y_proba)
    plot_confusion_matrix(y_test, y_pred)

    # Save
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "random_forest_model.joblib"))
    all_metrics = {**cv_metrics, **test_metrics, "best_params": best_params}
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("RANDOM FOREST COMPLETE")
    print(f"  CV AUC-ROC:   {cv_metrics['cv_auc_roc']:.4f}")
    print(f"  Test AUC-ROC: {test_metrics['test_auc_roc']:.4f}")
    print(f"  Test F1:      {test_metrics['test_f1']:.4f}")
    print(f"  Outputs: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
