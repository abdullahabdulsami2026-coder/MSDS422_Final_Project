"""
03d_mlp_neural_network.py
==========================
MLP Neural Network Model for Diabetes 30-Day Readmission Prediction

Deep learning approach using sklearn MLPClassifier with early stopping.

Author: Feifan
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

from sklearn.neural_network import MLPClassifier
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
OUTPUT_DIR = os.path.join("outputs", "mlp")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RANDOM_STATE = 42
N_FOLDS = 5


def load_processed_data():
    """Load preprocessed data (scaled — required for MLP)."""
    print("Loading preprocessed data (scaled)...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]:,} samples")
    print(f"  Positive rate (train): {y_train.mean():.3f}")
    return X_train, X_test, y_train, y_test


def tune_and_train(X_train, y_train):
    """Hyperparameter tuning for MLP."""
    print("\n--- Hyperparameter Tuning (GridSearchCV) ---")
    print("  Note: MLP tuning can be slow — using focused search space.")

    param_grid = {
        "hidden_layer_sizes": [
            (128, 64, 32),
            (100, 50),
            (256, 128, 64),
            (64, 32),
        ],
        "activation": ["relu"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["adaptive"],
        "learning_rate_init": [0.001, 0.01],
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    from sklearn.model_selection import RandomizedSearchCV

    grid = RandomizedSearchCV(
        MLPClassifier(
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
            solver="adam",
            batch_size=256,
        ),
        param_grid, n_iter=12, cv=cv, scoring="roc_auc",
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


def plot_training_loss(model):
    """Plot MLP training loss curve."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(model.loss_curve_, color="#10b981", linewidth=1.5, label="Training Loss")
    if hasattr(model, "validation_scores_") and model.validation_scores_:
        ax2 = ax.twinx()
        ax2.plot(model.validation_scores_, color="#3b82f6", linewidth=1.5,
                 label="Validation AUC", linestyle="--")
        ax2.set_ylabel("Validation Score", color="#3b82f6")
        ax2.legend(loc="center right")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss", color="#10b981")
    ax.set_title("MLP Training Loss Curve")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "mlp_training_loss.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curve(y_test, y_proba):
    """Plot ROC curve."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#10b981", linewidth=2, label=f"MLP (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#10b981")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — MLP Neural Network")
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
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Greens",
                xticklabels=["No Readmit", "Readmit <30d"],
                yticklabels=["No Readmit", "Readmit <30d"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — MLP Neural Network")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "confusion_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("MLP NEURAL NETWORK MODEL PIPELINE")
    print("Diabetes 30-Day Readmission Prediction")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_processed_data()

    best_model, best_params = tune_and_train(X_train, y_train)
    cv_metrics = cross_validate_model(best_model, X_train, y_train)
    test_metrics, y_pred, y_proba = evaluate_on_test(best_model, X_test, y_test)

    # Plots
    plot_training_loss(best_model)
    plot_roc_curve(y_test, y_proba)
    plot_confusion_matrix(y_test, y_pred)

    # Save
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "mlp_model.joblib"))
    all_metrics = {**cv_metrics, **test_metrics, "best_params": best_params}
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("MLP NEURAL NETWORK COMPLETE")
    print(f"  CV AUC-ROC:   {cv_metrics['cv_auc_roc']:.4f}")
    print(f"  Test AUC-ROC: {test_metrics['test_auc_roc']:.4f}")
    print(f"  Test F1:      {test_metrics['test_f1']:.4f}")
    print(f"  Outputs: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
