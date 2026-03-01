"""
05_model_comparison.py
======================
Model Comparison and Final Evaluation Dashboard

Loads metrics from all 4 models, creates:
- Comparison table (CSV + formatted console output)
- Combined ROC curves (all models on one plot)
- Combined Precision-Recall curves
- Performance bar charts
- Final recommendation summary

Author: Abdullah Abdulsami (comparison table), Jiahao Li (visualizations)
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
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
import joblib

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

# ============================================================
# Configuration
# ============================================================
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("outputs", "comparison")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

MODELS = {
    "Logistic Regression": {
        "metrics_path": os.path.join("outputs", "logistic_regression", "metrics.json"),
        "model_path": os.path.join("outputs", "logistic_regression", "logistic_regression_model.joblib"),
        "color": "#8b5cf6",
        "scaled": True,
        "owner": "Abdullah",
    },
    "Random Forest": {
        "metrics_path": os.path.join("outputs", "random_forest", "metrics.json"),
        "model_path": os.path.join("outputs", "random_forest", "random_forest_model.joblib"),
        "color": "#f59e0b",
        "scaled": False,
        "owner": "Abdullah",
    },
    "XGBoost": {
        "metrics_path": os.path.join("outputs", "xgboost", "metrics.json"),
        "model_path": os.path.join("outputs", "xgboost", "xgboost_best_model.joblib"),
        "color": "#2563eb",
        "scaled": False,
        "owner": "Jiahao",
    },
    "MLP Neural Network": {
        "metrics_path": os.path.join("outputs", "mlp", "metrics.json"),
        "model_path": os.path.join("outputs", "mlp", "mlp_model.joblib"),
        "color": "#10b981",
        "scaled": True,
        "owner": "Feifan",
    },
}


def load_test_data():
    """Load test data for generating curves."""
    X_test_scaled = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    X_test_unscaled = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_unscaled.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()
    return X_test_scaled, X_test_unscaled, y_test


def load_all_metrics():
    """Load metrics from each model's JSON output."""
    print("\n--- Loading Model Metrics ---")
    results = {}

    for name, config in MODELS.items():
        path = config["metrics_path"]
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
            print(f"  Loaded: {name}")
        else:
            print(f"  WARNING: {name} metrics not found at {path}")
            print(f"           Run the model script first.")

    return results


def create_comparison_table(results):
    """Create formatted comparison table."""
    print("\n--- Model Comparison Table ---\n")

    metrics_keys = [
        ("test_auc_roc", "AUC-ROC"),
        ("test_accuracy", "Accuracy"),
        ("test_precision", "Precision"),
        ("test_recall", "Recall"),
        ("test_f1", "F1 Score"),
        ("test_avg_precision", "Avg Precision"),
        ("cv_auc_roc", "CV AUC-ROC"),
        ("cv_f1", "CV F1"),
    ]

    rows = []
    for name, metrics in results.items():
        row = {"Model": name, "Owner": MODELS[name]["owner"]}
        for key, label in metrics_keys:
            row[label] = metrics.get(key, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by AUC-ROC descending
    df = df.sort_values("AUC-ROC", ascending=False).reset_index(drop=True)

    # Add rank
    df.insert(0, "Rank", range(1, len(df) + 1))

    # Print formatted
    print(df.to_string(index=False, float_format="%.4f"))

    # Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Identify best model
    best = df.iloc[0]
    print(f"\n  BEST MODEL: {best['Model']} (AUC-ROC: {best['AUC-ROC']:.4f})")

    return df


def plot_combined_roc(y_test, models_data):
    """Plot all model ROC curves on one figure."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    for name, data in models_data.items():
        if data["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, data["y_proba"])
            auc = roc_auc_score(y_test, data["y_proba"])
            ax.plot(fpr, tpr, color=MODELS[name]["color"], linewidth=2,
                    label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "combined_roc_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_combined_pr(y_test, models_data):
    """Plot all model Precision-Recall curves."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    for name, data in models_data.items():
        if data["y_proba"] is not None:
            precision, recall, _ = precision_recall_curve(y_test, data["y_proba"])
            ax.plot(recall, precision, color=MODELS[name]["color"], linewidth=2,
                    label=f"{name}")

    baseline = y_test.mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "combined_pr_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_metrics_barchart(comparison_df):
    """Bar chart comparing key metrics across models."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    metrics_to_plot = ["AUC-ROC", "F1 Score", "Precision", "Recall"]
    models = comparison_df["Model"].values
    colors = [MODELS[m]["color"] for m in models]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for idx, metric in enumerate(metrics_to_plot):
        values = comparison_df[metric].values
        bars = axes[idx].bar(range(len(models)), values, color=colors, width=0.6)
        axes[idx].set_xticks(range(len(models)))
        axes[idx].set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=8)
        axes[idx].set_title(metric, fontsize=12, fontweight="bold")
        axes[idx].set_ylim(0, max(values) * 1.2)
        axes[idx].spines["top"].set_visible(False)
        axes[idx].spines["right"].set_visible(False)

        # Value labels
        for bar, val in zip(bars, values):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    plt.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "metrics_comparison_barchart.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_summary(comparison_df, results):
    """Generate text summary for report."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    best = comparison_df.iloc[0]
    summary = []
    summary.append("=" * 60)
    summary.append("MODEL COMPARISON SUMMARY")
    summary.append("Predicting 30-Day Diabetes Readmission")
    summary.append("=" * 60)
    summary.append("")
    summary.append(f"Best Performing Model: {best['Model']}")
    summary.append(f"  Owner: {best['Owner']}")
    summary.append(f"  Test AUC-ROC: {best['AUC-ROC']:.4f}")
    summary.append(f"  Test F1 Score: {best['F1 Score']:.4f}")
    summary.append(f"  Test Precision: {best['Precision']:.4f}")
    summary.append(f"  Test Recall: {best['Recall']:.4f}")
    summary.append("")
    summary.append("All Models Ranked:")
    for _, row in comparison_df.iterrows():
        summary.append(
            f"  {row['Rank']}. {row['Model']:25s} | AUC: {row['AUC-ROC']:.4f} "
            f"| F1: {row['F1 Score']:.4f} | Owner: {row['Owner']}"
        )
    summary.append("")
    summary.append("Key Findings:")
    summary.append("- Class imbalance (11.2% positive) is a major challenge")
    summary.append("- AUC-ROC is the primary metric (robust to imbalance)")
    summary.append("- Gradient boosting methods handle sparse clinical features well")
    summary.append("- Feature importance reveals prior utilization as top predictor")
    summary.append("")
    summary.append("Recommendation:")
    summary.append(f"  Deploy {best['Model']} as the production model.")
    summary.append("  Monitor for data drift in medication patterns and")
    summary.append("  readmission definitions. Retrain quarterly.")
    summary.append("=" * 60)

    text = "\n".join(summary)
    print(text)

    path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"\n  Saved: {path}")


def main():
    print("=" * 60)
    print("MODEL COMPARISON PIPELINE")
    print("Diabetes 30-Day Readmission Prediction")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load metrics
    results = load_all_metrics()

    if len(results) == 0:
        print("\nERROR: No model metrics found. Run the individual model scripts first:")
        print("  python scripts/03a_logistic_regression.py")
        print("  python scripts/03b_random_forest.py")
        print("  python scripts/03c_xgboost_model.py")
        print("  python scripts/03d_mlp_neural_network.py")
        return

    # Comparison table
    comparison_df = create_comparison_table(results)

    # Load test data and models for curve plots
    print("\n--- Loading Models for Curve Generation ---")
    X_test_scaled, X_test_unscaled, y_test = load_test_data()

    models_data = {}
    for name, config in MODELS.items():
        if os.path.exists(config["model_path"]):
            model = joblib.load(config["model_path"])
            X = X_test_scaled if config["scaled"] else X_test_unscaled
            y_proba = model.predict_proba(X)[:, 1]
            models_data[name] = {"model": model, "y_proba": y_proba}
            print(f"  Loaded: {name}")
        else:
            models_data[name] = {"model": None, "y_proba": None}
            print(f"  SKIP: {name} (model file not found)")

    # Plots
    print("\n--- Generating Comparison Plots ---")
    plot_combined_roc(y_test, models_data)
    plot_combined_pr(y_test, models_data)
    plot_metrics_barchart(comparison_df)

    # Summary
    generate_summary(comparison_df, results)

    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON COMPLETE")
    print(f"  Outputs: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
