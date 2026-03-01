"""
03_xgboost_model.py
====================
XGBoost Model for Diabetes 30-Day Readmission Prediction

This script handles:
- Loading preprocessed data from feature engineering pipeline
- XGBoost model training with hyperparameter tuning
- Handling class imbalance via scale_pos_weight
- Cross-validation evaluation (AUC-ROC, Precision, Recall, F1)
- Feature importance analysis and visualization
- Model export for deployment

Author: Jiahao Li
Team: Final Project - G4 (Abdullah, Jiahao, Feifan)
Course: MSDS 422 - Practical Machine Learning
Date: February 2026

Dependencies:
    pip install xgboost scikit-learn pandas numpy matplotlib seaborn
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict,
)
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
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
OUTPUT_DIR = os.path.join("outputs", "xgboost")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RANDOM_STATE = 42
N_FOLDS = 5


def load_processed_data():
    """Load preprocessed train/test data."""
    print("Loading preprocessed data...")

    # Use unscaled data for XGBoost (tree-based models don't need scaling)
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_unscaled.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_unscaled.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]:,} samples")
    print(f"  Class balance (train): {y_train.mean():.3f} positive rate")

    return X_train, X_test, y_train, y_test


def compute_scale_pos_weight(y):
    """Calculate scale_pos_weight for class imbalance."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    weight = neg / pos
    print(f"  scale_pos_weight: {weight:.2f} (neg={neg:,}, pos={pos:,})")
    return weight


def train_xgboost_baseline(X_train, y_train, scale_weight):
    """Train baseline XGBoost with reasonable defaults."""
    print("\n--- Baseline XGBoost Training ---")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=scale_weight,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric="auc",
        use_label_encoder=False,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, verbose=False)
    print("  Baseline model trained successfully.")
    return model


def tune_hyperparameters(X_train, y_train, scale_weight):
    """
    Perform grid search for XGBoost hyperparameter tuning.

    Tuned parameters:
    - max_depth: tree depth (controls complexity)
    - learning_rate: step size shrinkage
    - n_estimators: number of boosting rounds
    - min_child_weight: minimum sum of instance weight in a child
    """
    print("\n--- Hyperparameter Tuning (GridSearchCV) ---")

    param_grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 300, 500],
        "min_child_weight": [1, 3, 5],
    }

    base_model = xgb.XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric="auc",
        use_label_encoder=False,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    print(f"\n  Best AUC-ROC (CV): {grid_search.best_score_:.4f}")
    print(f"  Best parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_, grid_search.best_params_


def cross_validate_model(model, X_train, y_train):
    """Perform stratified k-fold cross-validation."""
    print(f"\n--- {N_FOLDS}-Fold Cross-Validation ---")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Get cross-validated predictions
    y_pred_proba = cross_val_predict(
        model, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Metrics
    metrics = {
        "cv_auc_roc": roc_auc_score(y_train, y_pred_proba),
        "cv_accuracy": accuracy_score(y_train, y_pred),
        "cv_precision": precision_score(y_train, y_pred),
        "cv_recall": recall_score(y_train, y_pred),
        "cv_f1": f1_score(y_train, y_pred),
        "cv_avg_precision": average_precision_score(y_train, y_pred_proba),
    }

    print(f"  AUC-ROC:           {metrics['cv_auc_roc']:.4f}")
    print(f"  Accuracy:          {metrics['cv_accuracy']:.4f}")
    print(f"  Precision:         {metrics['cv_precision']:.4f}")
    print(f"  Recall:            {metrics['cv_recall']:.4f}")
    print(f"  F1 Score:          {metrics['cv_f1']:.4f}")
    print(f"  Avg Precision (PR):{metrics['cv_avg_precision']:.4f}")

    return metrics


def evaluate_on_test(model, X_test, y_test):
    """Evaluate final model on hold-out test set."""
    print("\n--- Test Set Evaluation ---")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "test_auc_roc": roc_auc_score(y_test, y_pred_proba),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_avg_precision": average_precision_score(y_test, y_pred_proba),
    }

    print(f"  AUC-ROC:           {metrics['test_auc_roc']:.4f}")
    print(f"  Accuracy:          {metrics['test_accuracy']:.4f}")
    print(f"  Precision:         {metrics['test_precision']:.4f}")
    print(f"  Recall:            {metrics['test_recall']:.4f}")
    print(f"  F1 Score:          {metrics['test_f1']:.4f}")
    print(f"  Avg Precision (PR):{metrics['test_avg_precision']:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Readmit", "Readmit <30d"]))

    return metrics, y_pred, y_pred_proba


# ============================================================
# Visualization Functions
# ============================================================

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot top-N feature importances from XGBoost.

    Creates three importance plots:
    1. Gain-based importance (default)
    2. Weight-based (frequency) importance
    3. Cover-based importance
    """
    print(f"\n--- Feature Importance Plots (Top {top_n}) ---")
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1. Gain-based importance (most informative)
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        range(len(feat_imp)),
        feat_imp["Importance"].values,
        color=sns.color_palette("viridis", len(feat_imp)),
    )
    ax.set_yticks(range(len(feat_imp)))
    ax.set_yticklabels(feat_imp["Feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("XGBoost Feature Importance — Top 20 Features (Gain)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels
    for bar, val in zip(bars, feat_imp["Importance"].values):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9
        )

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, "feature_importance_gain.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")

    # 2. XGBoost built-in importance plot (weight / frequency)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for idx, imp_type in enumerate(["weight", "cover"]):
        booster = model.get_booster()
        scores = booster.get_score(importance_type=imp_type)

        if scores:
            imp_df = (
                pd.DataFrame.from_dict(scores, orient="index", columns=["Importance"])
                .sort_values("Importance", ascending=False)
                .head(top_n)
            )
            # Map feature indices to names
            imp_df.index = [
                feature_names[int(f.replace("f", ""))]
                if f.startswith("f") and f[1:].isdigit()
                else f
                for f in imp_df.index
            ]

            axes[idx].barh(
                range(len(imp_df)),
                imp_df["Importance"].values,
                color=sns.color_palette("magma", len(imp_df))
                if idx == 0
                else sns.color_palette("cividis", len(imp_df)),
            )
            axes[idx].set_yticks(range(len(imp_df)))
            axes[idx].set_yticklabels(imp_df.index)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel(f"Importance ({imp_type.title()})")
            axes[idx].set_title(f"Feature Importance — {imp_type.title()}")
            axes[idx].spines["top"].set_visible(False)
            axes[idx].spines["right"].set_visible(False)

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, "feature_importance_weight_cover.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")

    # Save importance data to CSV
    feat_imp_full = pd.DataFrame({
        "Feature": feature_names,
        "Importance_Gain": importances,
    }).sort_values("Importance_Gain", ascending=False)
    csv_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    feat_imp_full.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return feat_imp


def plot_roc_curve(y_test, y_pred_proba):
    """Plot ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"XGBoost (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2563eb")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — XGBoost Readmission Prediction")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, "roc_curve.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_precision_recall_curve(y_test, y_pred_proba):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_prec = average_precision_score(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#dc2626", linewidth=2,
            label=f"XGBoost (AP = {avg_prec:.4f})")
    ax.axhline(y=y_test.mean(), color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline ({y_test.mean():.3f})")
    ax.fill_between(recall, precision, alpha=0.1, color="#dc2626")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — XGBoost")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, "precision_recall_curve.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["No Readmit", "Readmit <30d"],
        yticklabels=["No Readmit", "Readmit <30d"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — XGBoost")

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, "confusion_matrix.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_learning_curve(model, X_train, y_train):
    """Plot XGBoost training history with eval metric."""
    print("\n--- Learning Curve ---")

    eval_set = [(X_train, y_train)]
    model_lc = xgb.XGBClassifier(**model.get_params())
    model_lc.set_params(eval_metric="logloss")

    model_lc.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False,
    )

    results = model_lc.evals_result()
    epochs = range(len(results["validation_0"]["logloss"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, results["validation_0"]["logloss"], color="#2563eb", linewidth=1.5)
    ax.set_xlabel("Boosting Rounds")
    ax.set_ylabel("Log Loss")
    ax.set_title("XGBoost Training Loss Curve")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, "learning_curve.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# Main
# ============================================================

def main():
    """Run the full XGBoost pipeline."""
    print("=" * 60)
    print("XGBOOST MODEL PIPELINE")
    print("Diabetes 30-Day Readmission Prediction")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    feature_names = X_train.columns.tolist()

    # Class imbalance weight
    scale_weight = compute_scale_pos_weight(y_train)

    # Train baseline
    baseline_model = train_xgboost_baseline(X_train, y_train, scale_weight)

    # Hyperparameter tuning
    best_model, best_params = tune_hyperparameters(X_train, y_train, scale_weight)

    # Cross-validation
    cv_metrics = cross_validate_model(best_model, X_train, y_train)

    # Test evaluation
    test_metrics, y_pred, y_pred_proba = evaluate_on_test(best_model, X_test, y_test)

    # ---- Visualizations ----
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_feature_importance(best_model, feature_names, top_n=20)
    plot_roc_curve(y_test, y_pred_proba)
    plot_precision_recall_curve(y_test, y_pred_proba)
    plot_confusion_matrix(y_test, y_pred)
    plot_learning_curve(best_model, X_train, y_train)

    # ---- Save Model & Results ----
    print("\n" + "=" * 60)
    print("SAVING MODEL & RESULTS")
    print("=" * 60)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "xgboost_best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"  Model saved: {model_path}")

    # Save all metrics
    all_metrics = {**cv_metrics, **test_metrics, "best_params": best_params}
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"  Metrics saved: {metrics_path}")

    # Summary
    print("\n" + "=" * 60)
    print("XGBOOST PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Best Parameters: {best_params}")
    print(f"  CV AUC-ROC:   {cv_metrics['cv_auc_roc']:.4f}")
    print(f"  Test AUC-ROC: {test_metrics['test_auc_roc']:.4f}")
    print(f"  Test F1:      {test_metrics['test_f1']:.4f}")
    print(f"\n  All outputs saved to: {OUTPUT_DIR}/")
    print(f"  Plots saved to:       {PLOT_DIR}/")


if __name__ == "__main__":
    main()
