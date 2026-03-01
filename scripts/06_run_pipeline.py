"""
06_run_pipeline.py
==================
End-to-End Automated ML Pipeline Runner

Executes the complete workflow:
  1. Feature Engineering (preprocessing + feature creation)
  2. Model Training (all 4 models)
  3. Model Comparison (evaluation + visualization)
  4. Pipeline Diagram generation

Usage:
    python scripts/06_run_pipeline.py              # Run everything
    python scripts/06_run_pipeline.py --models-only # Skip preprocessing
    python scripts/06_run_pipeline.py --compare-only # Only comparison step

Author: Jiahao Li (pipeline), Abdullah (scripts), Feifan (documentation)
Team: Final Project - G4
Course: MSDS 422 - Practical Machine Learning
Date: February 2026
"""

import os
import sys
import time
import subprocess
import argparse
from datetime import datetime


SCRIPTS = {
    "Feature Engineering": "scripts/02_feature_engineering.py",
    "Logistic Regression": "scripts/03a_logistic_regression.py",
    "Random Forest": "scripts/03b_random_forest.py",
    "XGBoost": "scripts/03c_xgboost_model.py",
    "MLP Neural Network": "scripts/03d_mlp_neural_network.py",
    "Model Comparison": "scripts/05_model_comparison.py",
    "Pipeline Diagram": "scripts/04_pipeline_diagram.py",
}


def run_script(name, path):
    """Run a single Python script and report status."""
    print(f"\n{'─' * 60}")
    print(f"  RUNNING: {name}")
    print(f"  Script:  {path}")
    print(f"{'─' * 60}\n")

    if not os.path.exists(path):
        print(f"  ERROR: Script not found at {path}")
        return False, 0

    start = time.time()
    result = subprocess.run(
        [sys.executable, path],
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  SUCCESS: {name} completed in {elapsed:.1f}s")
        return True, elapsed
    else:
        print(f"\n  FAILED: {name} (exit code {result.returncode})")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run ML Pipeline")
    parser.add_argument("--models-only", action="store_true",
                        help="Skip preprocessing, run models + comparison only")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only run comparison step (models must be trained)")
    args = parser.parse_args()

    print("=" * 60)
    print("AUTOMATED ML PIPELINE")
    print("Predicting 30-Day Diabetes Readmission")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Determine which steps to run
    if args.compare_only:
        steps = ["Model Comparison"]
    elif args.models_only:
        steps = [
            "Logistic Regression", "Random Forest",
            "XGBoost", "MLP Neural Network",
            "Model Comparison", "Pipeline Diagram",
        ]
    else:
        steps = list(SCRIPTS.keys())

    # Check prerequisites
    data_file = os.path.join("data", "diabetic_data.csv")
    if not args.compare_only and not args.models_only:
        if not os.path.exists(data_file):
            print(f"\n  ERROR: Dataset not found at {data_file}")
            print("  Please download from UCI ML Repository first.")
            print("  See data/README.md for instructions.")
            sys.exit(1)

    if args.models_only or args.compare_only:
        processed = os.path.join("data", "processed", "X_train.csv")
        if not os.path.exists(processed):
            print(f"\n  ERROR: Processed data not found. Run full pipeline first.")
            sys.exit(1)

    # Run steps
    results = {}
    total_start = time.time()

    for step_name in steps:
        script_path = SCRIPTS[step_name]
        success, elapsed = run_script(step_name, script_path)
        results[step_name] = {"success": success, "time": elapsed}

        if not success and step_name == "Feature Engineering":
            print("\n  ABORTING: Preprocessing failed. Cannot continue.")
            sys.exit(1)

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"\n  {'Step':<25s} {'Status':<10s} {'Time':>8s}")
    print(f"  {'─' * 45}")

    for step_name, result in results.items():
        status = "PASS" if result["success"] else "FAIL"
        print(f"  {step_name:<25s} {status:<10s} {result['time']:>7.1f}s")

    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)

    print(f"\n  Total: {passed}/{total} steps passed")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n  Output directories:")
    print(f"    data/processed/           - Preprocessed datasets")
    print(f"    outputs/logistic_regression/ - LR model + plots")
    print(f"    outputs/random_forest/       - RF model + plots")
    print(f"    outputs/xgboost/             - XGBoost model + plots")
    print(f"    outputs/mlp/                 - MLP model + plots")
    print(f"    outputs/comparison/          - Comparison table + plots")
    print(f"    outputs/pipeline/            - Pipeline diagram")
    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
