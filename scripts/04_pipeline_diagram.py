"""
04_pipeline_diagram.py
======================
End-to-End ML Pipeline Diagram for Diabetes Readmission Prediction

Generates a professional pipeline diagram showing the full workflow
from data ingestion to model deployment.

Author: Jiahao Li
Team: Final Project - G4 (Abdullah, Jiahao, Feifan)
Course: MSDS 422 - Practical Machine Learning
Date: February 2026
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT_DIR = os.path.join("outputs", "pipeline")


def draw_pipeline_diagram():
    """Create the end-to-end ML pipeline diagram."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # Title
    ax.text(
        10, 13.5,
        "End-to-End ML Pipeline: Predicting 30-Day Diabetes Readmission",
        fontsize=18, fontweight="bold", ha="center", va="center",
        color="#1a1a2e",
    )
    ax.text(
        10, 13.0,
        "MSDS 422 Final Project — Team G4 (Abdullah, Jiahao, Feifan)",
        fontsize=11, ha="center", va="center", color="#555555",
    )

    # ================================================================
    # Color scheme
    # ================================================================
    colors = {
        "data": "#3b82f6",       # blue
        "preprocess": "#8b5cf6", # purple
        "feature": "#06b6d4",    # cyan
        "model": "#f59e0b",      # amber
        "eval": "#ef4444",       # red
        "deploy": "#10b981",     # green
        "arrow": "#374151",      # gray
    }

    def draw_box(x, y, w, h, color, title, items, alpha=0.15):
        """Draw a rounded box with title and bullet items."""
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, alpha=alpha,
            edgecolor=color, linewidth=2,
        )
        ax.add_patch(box)

        # Title
        ax.text(
            x + w / 2, y + h - 0.35,
            title, fontsize=12, fontweight="bold",
            ha="center", va="center", color=color,
        )

        # Items
        for i, item in enumerate(items):
            ax.text(
                x + 0.3, y + h - 0.75 - i * 0.35,
                f"• {item}", fontsize=8.5, va="center", color="#1a1a2e",
            )

    def draw_arrow(x1, y1, x2, y2, color="#374151"):
        """Draw a curved arrow between boxes."""
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=2,
            color=color,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    # ================================================================
    # Row 1: Data Ingestion -> Preprocessing -> Feature Engineering
    # ================================================================
    row1_y = 9.5
    box_h = 2.8
    box_w = 5.2

    # Box 1: Data Ingestion
    draw_box(0.5, row1_y, box_w, box_h, colors["data"],
             "1. DATA INGESTION", [
                 "UCI Diabetes 130-US Hospitals",
                 "101,766 encounters, 55 features",
                 "Target: readmission < 30 days",
                 "Binary classification (11.2% positive)",
                 "Source: UCI ML Repository",
                 "Format: CSV (diabetic_data.csv)",
             ])

    # Arrow 1->2
    draw_arrow(5.7 + 0.2, row1_y + box_h / 2, 7.4 - 0.2, row1_y + box_h / 2)

    # Box 2: Data Preprocessing
    draw_box(7.4, row1_y, box_w, box_h, colors["preprocess"],
             "2. DATA PREPROCESSING", [
                 "Remove duplicates (first encounter only)",
                 "Filter invalid discharges (deceased/hospice)",
                 "Drop: weight (97% missing), payer_code (52%)",
                 "Impute: race (mode), specialty ('Unknown')",
                 "ICD-9 diagnosis grouping (9 categories)",
                 "~70K encounters after cleaning",
             ])

    # Arrow 2->3
    draw_arrow(12.6 + 0.2, row1_y + box_h / 2, 14.3 - 0.2, row1_y + box_h / 2)

    # Box 3: Feature Engineering
    draw_box(14.3, row1_y, box_w, box_h, colors["feature"],
             "3. FEATURE ENGINEERING", [
                 "Length-of-stay bins (short/med/long/ext)",
                 "Total prior visits (OP + ER + IP)",
                 "Visit intensity categories",
                 "Emergency/inpatient visit ratios",
                 "Procedure intensity (per day)",
                 "High-utilizer flag (3+ prior visits)",
                 "22 new features → 77 total",
             ])

    # ================================================================
    # Row 2: Model Training (4 models) -> Evaluation
    # ================================================================
    row2_y = 4.5
    model_w = 3.5
    model_h = 3.8

    # Arrow down from Feature Engineering to Models
    draw_arrow(17, row1_y - 0.1, 17, row2_y + model_h + 0.1)
    ax.text(17.5, row2_y + model_h + 0.5, "Train/Test\nSplit (80/20)",
            fontsize=8, ha="center", va="center", color="#555555",
            style="italic")

    # Box 4a: Logistic Regression (Abdullah)
    draw_box(0.3, row2_y, model_w, model_h, colors["model"],
             "4a. LOGISTIC REG.", [
                 "Owner: Abdullah",
                 "L2 regularization",
                 "Class weight: balanced",
                 "Baseline model",
                 "Interpretable coefficients",
                 "Scaled features required",
                 "Threshold optimization",
             ])

    # Box 4b: Random Forest (Abdullah)
    draw_box(4.1, row2_y, model_w, model_h, colors["model"],
             "4b. RANDOM FOREST", [
                 "Owner: Abdullah",
                 "n_estimators: 300",
                 "max_depth tuned (CV)",
                 "Class weight: balanced",
                 "Ensemble of decision trees",
                 "Built-in feature importance",
                 "No scaling needed",
             ])

    # Box 4c: XGBoost (Jiahao)
    draw_box(7.9, row2_y, model_w, model_h, colors["model"],
             "4c. XGBOOST ★", [
                 "Owner: Jiahao",
                 "Gradient boosting framework",
                 "scale_pos_weight for imbalance",
                 "GridSearchCV tuning:",
                 "  depth, lr, n_est, min_child",
                 "5-fold stratified CV",
                 "Regularization (L1 + L2)",
             ])

    # Box 4d: MLP Neural Network (Feifan)
    draw_box(11.7, row2_y, model_w, model_h, colors["model"],
             "4d. MLP (NEURAL NET)", [
                 "Owner: Feifan",
                 "Multi-layer perceptron",
                 "Hidden layers: (128, 64, 32)",
                 "ReLU activation + Dropout",
                 "Adam optimizer",
                 "Early stopping",
                 "Scaled features required",
             ])

    # Box 5: Evaluation
    draw_box(15.7, row2_y, 3.8, model_h, colors["eval"],
             "5. EVALUATION", [
                 "AUC-ROC (primary metric)",
                 "Precision / Recall / F1",
                 "Confusion matrix",
                 "ROC & PR curves",
                 "5-fold cross-validation",
                 "Model comparison table",
                 "Feature importance analysis",
             ])

    # Arrows from models to evaluation
    for x_start in [2.1, 5.9, 9.7, 13.5]:
        draw_arrow(x_start + model_w / 2, row2_y - 0.05,
                    17.6, row2_y - 0.05, color="#999999")

    # ================================================================
    # Row 3: Deployment
    # ================================================================
    row3_y = 1.0
    deploy_w = 18.5
    deploy_h = 2.2

    # Arrow down from evaluation
    draw_arrow(17.6, row2_y - 0.1, 10, row3_y + deploy_h + 0.15)

    draw_box(0.5, row3_y, deploy_w, deploy_h, colors["deploy"],
             "6. DEPLOYMENT & AUTOMATION", [
                 "Best model selection based on AUC-ROC  →  Export via joblib  →  "
                 "Automated pipeline (preprocessing → prediction)  →  "
                 "Dashboard (model comparison, feature importance, demographics)  →  "
                 "Monitoring & retraining schedule",
             ])

    # Additional deployment details
    ax.text(
        1.0, row3_y + 0.5,
        "Pipeline: Raw CSV → clean → engineer features → predict → output risk scores    |    "
        "Tools: Python, scikit-learn, XGBoost, pandas, matplotlib    |    "
        "Infra: GitHub repo, requirements.txt, automated scripts",
        fontsize=8, color="#555555", va="center",
    )

    # ================================================================
    # Legend
    # ================================================================
    legend_items = [
        mpatches.Patch(facecolor=colors["data"], alpha=0.3, label="Data Ingestion"),
        mpatches.Patch(facecolor=colors["preprocess"], alpha=0.3, label="Preprocessing"),
        mpatches.Patch(facecolor=colors["feature"], alpha=0.3, label="Feature Engineering"),
        mpatches.Patch(facecolor=colors["model"], alpha=0.3, label="Model Training"),
        mpatches.Patch(facecolor=colors["eval"], alpha=0.3, label="Evaluation"),
        mpatches.Patch(facecolor=colors["deploy"], alpha=0.3, label="Deployment"),
    ]
    ax.legend(
        handles=legend_items, loc="upper right",
        fontsize=9, framealpha=0.9, ncol=3,
        bbox_to_anchor=(0.98, 0.98),
    )

    plt.tight_layout()

    # Save
    filepath = os.path.join(OUTPUT_DIR, "ml_pipeline_diagram.png")
    plt.savefig(filepath, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Pipeline diagram saved: {filepath}")

    # Also save as PDF for report
    filepath_pdf = os.path.join(OUTPUT_DIR, "ml_pipeline_diagram.pdf")
    fig2, ax2 = plt.subplots(figsize=(20, 14))
    ax2.axis("off")
    # Re-render for PDF (reuse the same function structure)
    plt.close()

    return filepath


def main():
    """Generate pipeline diagram."""
    print("=" * 60)
    print("GENERATING ML PIPELINE DIAGRAM")
    print("=" * 60)
    filepath = draw_pipeline_diagram()
    print(f"\nDone! Output: {filepath}")
    print("Upload this to your GitHub repo under outputs/ or docs/")


if __name__ == "__main__":
    main()
