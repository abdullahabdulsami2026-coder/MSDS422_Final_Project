"""
02_feature_engineering.py
=========================
Feature Engineering for Diabetes 130-US Hospitals Readmission Prediction

This script handles data preprocessing and feature engineering, including:
- Data cleaning and filtering
- Missing value handling
- Time-based feature creation (length of stay bins, prior visit counts)
- Categorical encoding
- Feature scaling
- Train/test split and export

Author: Jiahao Li
Team: Final Project - G4 (Abdullah, Jiahao, Feifan)
Course: MSDS 422 - Practical Machine Learning
Date: February 2026
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
RAW_DATA_PATH = os.path.join("data", "diabetic_data.csv")
OUTPUT_DIR = os.path.join("data", "processed")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw_data(filepath):
    """Load the raw UCI diabetes dataset."""
    print(f"Loading raw data from {filepath}...")
    df = pd.read_csv(filepath, na_values="?")
    print(f"  Raw dataset shape: {df.shape}")
    print(f"  Encounters: {len(df):,}")
    print(f"  Features: {df.shape[1]}")
    return df


def clean_data(df):
    """
    Clean dataset by removing invalid records and duplicates.

    Steps:
    1. Remove duplicate patient encounters (keep first)
    2. Remove deceased/hospice discharges (not readmission candidates)
    3. Drop high-missing-value columns (weight, payer_code)
    """
    print("\n--- Data Cleaning ---")
    initial_count = len(df)

    # Keep only the first encounter per patient
    df = df.sort_values("encounter_id").drop_duplicates(
        subset=["patient_nbr"], keep="first"
    )
    print(f"  After dedup (first encounter per patient): {len(df):,} "
          f"(removed {initial_count - len(df):,})")

    # Remove deceased/hospice discharges (discharge_disposition_id in [11,13,14,19,20,21])
    invalid_discharges = [11, 13, 14, 19, 20, 21]
    before = len(df)
    df = df[~df["discharge_disposition_id"].isin(invalid_discharges)]
    print(f"  After removing invalid discharges: {len(df):,} "
          f"(removed {before - len(df):,})")

    # Drop columns with excessive missing values
    drop_cols = ["weight", "payer_code", "encounter_id", "patient_nbr"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"  Dropped columns: {[c for c in drop_cols if c != 'encounter_id']}")

    print(f"  Final cleaned shape: {df.shape}")
    return df


def create_target_variable(df):
    """Create binary readmission target variable."""
    print("\n--- Target Variable ---")
    df["readmit_binary"] = (df["readmitted"] == "<30").astype(int)

    pos = df["readmit_binary"].sum()
    neg = len(df) - pos
    print(f"  Positive class (readmitted <30 days): {pos:,} ({pos/len(df)*100:.1f}%)")
    print(f"  Negative class: {neg:,} ({neg/len(df)*100:.1f}%)")
    print(f"  Imbalance ratio: {neg/pos:.1f}:1")

    df = df.drop(columns=["readmitted"])
    return df


def engineer_time_based_features(df):
    """
    Create time-based and utilization features.

    Features created:
    - los_bin: Length of stay bins (short/medium/long/extended)
    - total_prior_visits: Sum of outpatient + emergency + inpatient visits
    - prior_visit_intensity: Categorical intensity of prior visits
    - visit_ratio_emergency: Proportion of emergency visits among prior visits
    - visit_ratio_inpatient: Proportion of inpatient visits among prior visits
    - num_medications_bin: Medication count bins
    - high_utilizer: Flag for patients with 3+ prior visits
    - procedure_intensity: Ratio of procedures to time in hospital
    """
    print("\n--- Time-Based & Utilization Features ---")

    # Length of stay bins
    df["los_bin"] = pd.cut(
        df["time_in_hospital"],
        bins=[0, 2, 5, 9, 14],
        labels=["short", "medium", "long", "extended"],
        right=True
    )
    print(f"  los_bin distribution:\n{df['los_bin'].value_counts().to_string()}")

    # Total prior visits (outpatient + emergency + inpatient)
    df["total_prior_visits"] = (
        df["number_outpatient"]
        + df["number_emergency"]
        + df["number_inpatient"]
    )

    # Prior visit intensity categories
    df["prior_visit_intensity"] = pd.cut(
        df["total_prior_visits"],
        bins=[-1, 0, 2, 5, 999],
        labels=["none", "low", "moderate", "high"]
    )

    # Visit type ratios (proportion of emergency and inpatient among total)
    df["visit_ratio_emergency"] = np.where(
        df["total_prior_visits"] > 0,
        df["number_emergency"] / df["total_prior_visits"],
        0
    )
    df["visit_ratio_inpatient"] = np.where(
        df["total_prior_visits"] > 0,
        df["number_inpatient"] / df["total_prior_visits"],
        0
    )

    # Number of medications bin
    df["num_medications_bin"] = pd.cut(
        df["num_medications"],
        bins=[0, 5, 10, 20, 81],
        labels=["low", "moderate", "high", "very_high"],
        right=True
    )

    # High utilizer flag (3+ prior visits in the past year)
    df["high_utilizer"] = (df["total_prior_visits"] >= 3).astype(int)

    # Procedure intensity (procedures per day of stay)
    df["procedure_intensity"] = np.where(
        df["time_in_hospital"] > 0,
        df["num_procedures"] / df["time_in_hospital"],
        0
    )

    # Number of diagnoses interaction with prior visits
    df["diagnoses_x_visits"] = df["number_diagnoses"] * df["total_prior_visits"]

    # Lab procedures per day
    df["lab_per_day"] = np.where(
        df["time_in_hospital"] > 0,
        df["num_lab_procedures"] / df["time_in_hospital"],
        0
    )

    new_features = [
        "los_bin", "total_prior_visits", "prior_visit_intensity",
        "visit_ratio_emergency", "visit_ratio_inpatient",
        "num_medications_bin", "high_utilizer", "procedure_intensity",
        "diagnoses_x_visits", "lab_per_day"
    ]
    print(f"\n  Created {len(new_features)} new features: {new_features}")
    return df


def handle_missing_values(df):
    """Handle missing values in the dataset."""
    print("\n--- Missing Value Handling ---")

    # Race: impute with mode
    if df["race"].isna().sum() > 0:
        mode_val = df["race"].mode()[0]
        n_missing = df["race"].isna().sum()
        df["race"] = df["race"].fillna(mode_val)
        print(f"  race: {n_missing} missing -> imputed with mode ('{mode_val}')")

    # Medical specialty: fill with 'Unknown'
    if "medical_specialty" in df.columns:
        n_missing = df["medical_specialty"].isna().sum()
        df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
        print(f"  medical_specialty: {n_missing} missing -> filled with 'Unknown'")

    # Diagnosis codes: fill missing with 'Unknown'
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns and df[col].isna().sum() > 0:
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna("Unknown")
            print(f"  {col}: {n_missing} missing -> filled with 'Unknown'")

    # Any remaining missing in categorical columns
    remaining = df.isna().sum()
    remaining = remaining[remaining > 0]
    if len(remaining) > 0:
        print(f"\n  Remaining missing values:")
        for col, count in remaining.items():
            if df[col].dtype == "object" or str(df[col].dtype) == "category":
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(df[col].median())
            print(f"    {col}: {count} -> imputed")

    print(f"  Total missing after handling: {df.isna().sum().sum()}")
    return df


def encode_categorical_features(df):
    """
    Encode categorical features for model training.

    - Binary medication columns: map to numeric
    - Ordinal features: label encode
    - High-cardinality categoricals: target encode or drop
    """
    print("\n--- Categorical Encoding ---")

    # Medication columns: map dosage changes to numeric
    med_columns = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "examide",
        "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone"
    ]
    med_map = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}

    for col in med_columns:
        if col in df.columns:
            df[col] = df[col].map(med_map).fillna(0).astype(int)

    # Binary columns
    binary_map = {"No": 0, "Yes": 1, "Ch": 1}
    for col in ["change", "diabetesMed"]:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0).astype(int)

    # Age: convert to ordinal
    age_map = {
        "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3,
        "[40-50)": 4, "[50-60)": 5, "[60-70)": 6, "[70-80)": 7,
        "[80-90)": 8, "[90-100)": 9
    }
    if "age" in df.columns:
        df["age"] = df["age"].map(age_map).fillna(5).astype(int)

    # A1Cresult and max_glu_serum: ordinal encode
    a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
    glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
    if "A1Cresult" in df.columns:
        df["A1Cresult"] = df["A1Cresult"].map(a1c_map).fillna(0).astype(int)
    if "max_glu_serum" in df.columns:
        df["max_glu_serum"] = df["max_glu_serum"].map(glu_map).fillna(0).astype(int)

    # Gender: binary encode
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 0, "Male": 1}).fillna(0).astype(int)

    # Drop high-cardinality string columns that would need complex encoding
    # (medical_specialty, diag_1, diag_2, diag_3 -> use grouped versions or drop)
    high_card_cols = ["medical_specialty", "diag_1", "diag_2", "diag_3"]
    for col in high_card_cols:
        if col in df.columns:
            # Create a simplified grouping for diagnosis codes
            if col.startswith("diag_"):
                df[col + "_group"] = df[col].apply(_group_diagnosis)
                df = df.drop(columns=[col])
            else:
                # Medical specialty: group into top categories
                top_specs = df[col].value_counts().head(10).index.tolist()
                df[col] = df[col].apply(
                    lambda x: x if x in top_specs else "Other"
                )

    # Label encode remaining object columns
    le = LabelEncoder()
    object_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in object_cols:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        print(f"  Label encoded: {col} ({df[col].nunique()} categories)")

    print(f"\n  Final encoded shape: {df.shape}")
    return df


def _group_diagnosis(code):
    """Group ICD-9 codes into broad categories."""
    if code == "Unknown" or pd.isna(code):
        return "Other"
    try:
        num = float(code)
    except (ValueError, TypeError):
        # E or V codes
        if str(code).startswith("E"):
            return "External_Causes"
        elif str(code).startswith("V"):
            return "Supplementary"
        return "Other"

    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    elif 460 <= num <= 519 or num == 786:
        return "Respiratory"
    elif 520 <= num <= 579 or num == 787:
        return "Digestive"
    elif 250 <= num < 251:
        return "Diabetes"
    elif 800 <= num <= 999:
        return "Injury"
    elif 710 <= num <= 739:
        return "Musculoskeletal"
    elif 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    elif 140 <= num <= 239:
        return "Neoplasms"
    else:
        return "Other"


def scale_features(X_train, X_test):
    """Apply standard scaling to numeric features."""
    print("\n--- Feature Scaling ---")
    scaler = StandardScaler()
    feature_names = X_train.columns

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names,
        index=X_test.index
    )

    print(f"  Scaled {len(feature_names)} features")
    return X_train_scaled, X_test_scaled, scaler


def main():
    """Run the full feature engineering pipeline."""
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("Diabetes 30-Day Readmission Prediction")
    print("=" * 60)

    # Load
    df = load_raw_data(RAW_DATA_PATH)

    # Clean
    df = clean_data(df)

    # Target variable
    df = create_target_variable(df)

    # Feature engineering
    df = engineer_time_based_features(df)

    # Missing values
    df = handle_missing_values(df)

    # Encode
    df = encode_categorical_features(df)

    # Split features and target
    target = "readmit_binary"
    X = df.drop(columns=[target])
    y = df[target]

    print(f"\n--- Final Dataset ---")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]:,}")

    # Train/test split (stratified due to class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train set: {X_train.shape[0]:,}")
    print(f"  Test set: {X_test.shape[0]:,}")

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Save processed data
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train_scaled.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

    # Also save unscaled for tree-based models (XGBoost doesn't need scaling)
    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train_unscaled.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test_unscaled.csv"), index=False)

    # Save feature names
    with open(os.path.join(OUTPUT_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(X.columns.tolist()))

    print(f"\n  Saved processed data to {OUTPUT_DIR}/")
    print(f"  Files: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print(f"         X_train_unscaled.csv, X_test_unscaled.csv")
    print(f"         feature_names.txt")
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
