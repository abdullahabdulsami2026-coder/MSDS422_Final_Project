"""
MSDS 422 Final Project - Milestone 2
Exploratory Data Analysis
Hospital Readmission Prediction for Diabetes Patients

Dataset: Diabetes 130-US Hospitals (UCI Machine Learning Repository)
https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008

Author: [Your Name]
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. LOAD DATA
# ============================================================================

def load_data(filepath='diabetic_data.csv'):
    """
    Load the diabetes readmission dataset
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    df : DataFrame
        Loaded dataset
    """
    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    
    df = pd.read_csv(filepath, na_values='?')
    
    print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df


# ============================================================================
# 2. DATA OVERVIEW
# ============================================================================

def data_overview(df):
    """Provide comprehensive overview of the dataset"""
    
    print("\n" + "="*80)
    print("DATA OVERVIEW")
    print("="*80)
    
    # Basic info
    print("\n### DataFrame Info ###")
    df.info()
    
    # Data types
    print("\n### Data Types ###")
    print(df.dtypes.value_counts())
    
    # Memory usage
    print(f"\n### Memory Usage ###")
    print(f"Total memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return None


# ============================================================================
# 3. MISSING DATA ANALYSIS
# ============================================================================

def analyze_missing_data(df):
    """
    Analyze and visualize missing data patterns
    """
    print("\n" + "="*80)
    print("MISSING DATA ANALYSIS")
    print("="*80)
    
    # Calculate missing percentages
    missing = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df)) * 100
    })
    
    missing = missing[missing['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    
    if len(missing) > 0:
        print("\n### Columns with Missing Data ###")
        print(missing.to_string(index=False))
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(missing['column'], missing['missing_pct'])
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_title('Missing Data by Feature')
        ax.axvline(x=50, color='red', linestyle='--', label='50% threshold')
        ax.legend()
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/01_missing_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n✓ Saved: 01_missing_data_analysis.png")
    else:
        print("\nNo missing data found!")
    
    return missing


# ============================================================================
# 4. TARGET VARIABLE ANALYSIS
# ============================================================================

def analyze_target_variable(df, target_col='readmitted'):
    """
    Analyze the distribution of the target variable
    """
    print("\n" + "="*80)
    print("TARGET VARIABLE ANALYSIS")
    print("="*80)
    
    # Create binary target: <30 days vs. others
    df['readmit_binary'] = (df[target_col] == '<30').astype(int)
    
    # Value counts
    print(f"\n### Original Target Distribution ###")
    print(df[target_col].value_counts())
    print(f"\nPercentages:")
    print(df[target_col].value_counts(normalize=True) * 100)
    
    print(f"\n### Binary Target Distribution (Readmitted <30 days) ###")
    print(df['readmit_binary'].value_counts())
    print(f"\nClass Imbalance Ratio: {df['readmit_binary'].value_counts()[0] / df['readmit_binary'].value_counts()[1]:.2f}:1")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original distribution
    df[target_col].value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Original Readmission Distribution')
    axes[0].set_ylabel('Count')
    axes[0].set_xlabel('Readmission Status')
    
    # Binary distribution
    labels = ['Not Readmitted (<30 days)', 'Readmitted (<30 days)']
    colors = ['lightgreen', 'salmon']
    df['readmit_binary'].value_counts().plot(kind='pie', ax=axes[1], 
                                              labels=labels, colors=colors, autopct='%1.1f%%')
    axes[1].set_title('Binary Readmission Distribution')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/02_target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Saved: 02_target_distribution.png")
    
    return df


# ============================================================================
# 5. DEMOGRAPHIC ANALYSIS
# ============================================================================

def analyze_demographics(df):
    """
    Analyze demographic variables
    """
    print("\n" + "="*80)
    print("DEMOGRAPHIC ANALYSIS")
    print("="*80)
    
    demographic_vars = ['age', 'gender', 'race']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, var in enumerate(demographic_vars):
        if var in df.columns:
            print(f"\n### {var.upper()} Distribution ###")
            print(df[var].value_counts())
            
            df[var].value_counts().plot(kind='bar', ax=axes[idx], color='steelblue')
            axes[idx].set_title(f'{var.capitalize()} Distribution')
            axes[idx].set_ylabel('Count')
            axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/03_demographics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Saved: 03_demographics.png")
    
    return None


# ============================================================================
# 6. CONTINUOUS VARIABLES ANALYSIS
# ============================================================================

def analyze_continuous_variables(df):
    """
    Analyze continuous/numeric variables
    """
    print("\n" + "="*80)
    print("CONTINUOUS VARIABLES ANALYSIS")
    print("="*80)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove encounter and patient IDs
    numeric_cols = [col for col in numeric_cols if not any(x in col.lower() for x in ['id', 'number'])]
    
    if len(numeric_cols) > 0:
        print(f"\n### Summary Statistics ###")
        print(df[numeric_cols].describe().T)
        
        # Create distribution plots
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            if idx < len(axes):
                df[col].hist(bins=30, ax=axes[idx], color='teal', edgecolor='black')
                axes[idx].set_title(f'{col} Distribution')
                axes[idx].set_ylabel('Frequency')
        
        # Hide empty subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/04_continuous_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n✓ Saved: 04_continuous_distributions.png")
    
    return numeric_cols


# ============================================================================
# 7. MEDICATION ANALYSIS
# ============================================================================

def analyze_medications(df):
    """
    Analyze diabetes medications
    """
    print("\n" + "="*80)
    print("MEDICATION ANALYSIS")
    print("="*80)
    
    # Identify medication columns
    med_cols = [col for col in df.columns if any(x in col.lower() for x in 
                ['insulin', 'metformin', 'glipizide', 'glyburide', 'pioglitazone',
                 'rosiglitazone', 'acarbose', 'miglitol', 'chlorpropamide', 
                 'glimepiride', 'tolbutamide', 'tolazamide', 'nateglinide',
                 'repaglinide', 'troglitazone'])]
    
    if len(med_cols) > 0:
        print(f"\nFound {len(med_cols)} medication columns")
        
        # Count changes per medication
        med_changes = {}
        for col in med_cols:
            if col in df.columns:
                changes = df[col].value_counts()
                med_changes[col] = changes.get('Up', 0) + changes.get('Down', 0)
        
        med_changes_df = pd.DataFrame(list(med_changes.items()), 
                                      columns=['Medication', 'Changes'])
        med_changes_df = med_changes_df.sort_values('Changes', ascending=False).head(15)
        
        print("\n### Top 15 Medications by Dosage Changes ###")
        print(med_changes_df.to_string(index=False))
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(med_changes_df['Medication'], med_changes_df['Changes'], color='coral')
        ax.set_xlabel('Number of Dosage Changes')
        ax.set_title('Top 15 Diabetes Medications by Dosage Changes')
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/05_medication_changes.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n✓ Saved: 05_medication_changes.png")
    
    return med_cols


# ============================================================================
# 8. FEATURE CORRELATION ANALYSIS
# ============================================================================

def analyze_correlations(df, numeric_cols):
    """
    Analyze correlations between numeric features
    """
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    if 'readmit_binary' in df.columns:
        numeric_cols_with_target = numeric_cols + ['readmit_binary']
    else:
        numeric_cols_with_target = numeric_cols
    
    # Compute correlation matrix
    corr_matrix = df[numeric_cols_with_target].corr()
    
    # Find high correlations with target
    if 'readmit_binary' in corr_matrix.columns:
        target_corr = corr_matrix['readmit_binary'].drop('readmit_binary').sort_values(ascending=False)
        print("\n### Correlations with Readmission (Binary) ###")
        print(target_corr)
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/06_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Saved: 06_correlation_matrix.png")
    
    return corr_matrix


# ============================================================================
# 9. CATEGORICAL FEATURES VS TARGET
# ============================================================================

def analyze_categorical_vs_target(df):
    """
    Analyze categorical features against readmission
    """
    print("\n" + "="*80)
    print("CATEGORICAL FEATURES VS TARGET")
    print("="*80)
    
    categorical_vars = ['admission_type_id', 'discharge_disposition_id', 
                       'admission_source_id', 'medical_specialty']
    
    existing_vars = [var for var in categorical_vars if var in df.columns]
    
    if len(existing_vars) > 0 and 'readmit_binary' in df.columns:
        for var in existing_vars:
            print(f"\n### Readmission Rate by {var} ###")
            readmit_by_var = df.groupby(var)['readmit_binary'].agg(['sum', 'count', 'mean'])
            readmit_by_var.columns = ['Readmitted', 'Total', 'Rate']
            readmit_by_var = readmit_by_var.sort_values('Rate', ascending=False).head(10)
            print(readmit_by_var)
    
    return None


# ============================================================================
# 10. MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("\n" + "="*80)
    print("DIABETES READMISSION EDA - MILESTONE 2")
    print("="*80)
    
    # Load data
    df = load_data('diabetic_data.csv')
    
    # Data overview
    data_overview(df)
    
    # Missing data
    missing_data = analyze_missing_data(df)
    
    # Target variable
    df = analyze_target_variable(df)
    
    # Demographics
    analyze_demographics(df)
    
    # Continuous variables
    numeric_cols = analyze_continuous_variables(df)
    
    # Medications
    med_cols = analyze_medications(df)
    
    # Correlations
    if len(numeric_cols) > 0:
        corr_matrix = analyze_correlations(df, numeric_cols)
    
    # Categorical vs target
    analyze_categorical_vs_target(df)
    
    print("\n" + "="*80)
    print("EDA COMPLETE!")
    print("="*80)
    print("\nAll visualizations saved to /mnt/user-data/outputs/")
    print("\nNext steps:")
    print("1. Review all generated plots")
    print("2. Identify features for engineering")
    print("3. Decide on handling missing data")
    print("4. Plan preprocessing pipeline")
    
    return df


if __name__ == "__main__":
    df = main()
