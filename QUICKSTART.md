# Quick Start Guide - MSDS 422 Final Project

## 🚀 Getting Started in 5 Minutes

### Step 1: Clone or Download This Repository
```bash
# If you have git
git clone https://github.com/YOUR_USERNAME/MSDS422_Diabetes_Readmission_Project.git
cd MSDS422_Diabetes_Readmission_Project

# Or download ZIP from GitHub and extract
```

### Step 2: Set Up Python Environment
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Download Dataset
1. Go to: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
2. Download `dataset_diabetes.zip`
3. Extract `diabetic_data.csv` to the `data/` folder

### Step 4: Run Exploratory Data Analysis
```bash
cd scripts
python 01_exploratory_data_analysis.py
```

**Expected Output:**
- Console prints showing data shape, missing values, target distribution
- 6 PNG figures created in `outputs/figures/`:
  - 01_missing_data_analysis.png
  - 02_target_distribution.png
  - 03_demographics.png
  - 04_continuous_distributions.png
  - 05_medication_changes.png
  - 06_correlation_matrix.png

### Step 5: View Results
```bash
cd ../outputs/figures
ls -l
# Open the PNG files to see the visualizations
```

## 📂 Repository Structure

```
MSDS422_Diabetes_Readmission_Project/
├── README.md                    ← Start here for full documentation
├── QUICKSTART.md               ← This file
├── requirements.txt            ← Python dependencies
├── .gitignore                  ← Git ignore rules
├── data/
│   ├── README.md              ← Dataset download instructions
│   └── diabetic_data.csv      ← (You download this)
├── scripts/
│   └── 01_exploratory_data_analysis.py  ← Run this first
├── notebooks/
│   └── .gitkeep
├── outputs/
│   └── figures/
│       └── .gitkeep
├── docs/
│   ├── Milestone2_Report_FINAL.docx          ← Main project proposal
│   └── Annotated_Bibliography_FINAL.docx     ← 9 paper annotations
└── presentation/
    └── .gitkeep
```

## 📊 What This Project Does

**Goal:** Predict which diabetes patients will be readmitted to the hospital within 30 days of discharge.

**Dataset:** 101,766 hospital encounters → ~70,000 after cleaning

**Models:** Logistic Regression, Random Forest, XGBoost, Deep Neural Network

**Target Performance:** AUROC ≥0.75 (beat literature benchmark of 0.667)

## 📝 Milestone 2 Deliverables (Current Status)

✅ **docs/Milestone2_Report_FINAL.docx** - Complete project proposal  
✅ **docs/Annotated_Bibliography_FINAL.docx** - 9 papers (6 complete, 3 need completion)  
✅ **scripts/01_exploratory_data_analysis.py** - EDA code  
⏳ **outputs/figures/** - 6 charts (generated when you run EDA script)  
⏳ **presentation/Milestone2_Slides.pptx** - To be created  

## ⚠️ TODO Before Submission

1. **Complete Annotated Bibliography** - Open `docs/Annotated_Bibliography_FINAL.docx`
   - Abdullah: Write Paper #7 annotation (search for "[TO BE COMPLETED")
   - Feifan: Write Paper #8 annotation
   - Jiahao: Write Paper #9 annotation

2. **Run EDA Script** - Generate the 6 figures:
   ```bash
   cd scripts
   python 01_exploratory_data_analysis.py
   ```

3. **Create Presentation** - Make PowerPoint with 10-12 slides:
   - Title, Executive Summary, Problem Statement, Dataset Overview
   - EDA Findings (use your generated charts)
   - Feature Engineering Plan, Model Strategy
   - Expected Results, Deployment Plan, Next Steps, Team Contributions

4. **Push to GitHub** - Upload everything to your GitHub account

5. **Submit to Canvas** - Submit your GitHub repository link

## 🆘 Common Issues

**Issue:** `ModuleNotFoundError: No module named 'pandas'`  
**Fix:** Make sure you ran `pip install -r requirements.txt`

**Issue:** `FileNotFoundError: diabetic_data.csv`  
**Fix:** Download dataset from UCI (see data/README.md)

**Issue:** Charts don't appear  
**Fix:** Check `outputs/figures/` folder - they're saved as PNG files

**Issue:** Git won't upload large CSV file  
**Fix:** Don't upload the dataset! It's in .gitignore - just link to UCI repository

## 📧 Questions?

Review the full documentation in `README.md` for detailed methodology, features, and deployment strategy.

---

**Last Updated:** February 8, 2026  
**Team:** Abdullah Abdul Sami, Feifan Liu, Jiahao Li
