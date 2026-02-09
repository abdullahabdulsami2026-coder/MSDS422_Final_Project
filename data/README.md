# Dataset Information

## UCI Diabetes 130-US Hospitals Dataset

### Download Instructions

1. **Visit the UCI Machine Learning Repository:**
   https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008

2. **Download the dataset:**
   - Click on "Data Folder" link
   - Download `dataset_diabetes.zip` (approximately 20 MB)

3. **Extract the files:**
   ```bash
   unzip dataset_diabetes.zip
   ```

4. **Copy to this directory:**
   ```bash
   cp diabetic_data.csv /path/to/MSDS422_Diabetes_Readmission_Project/data/
   ```

### Dataset Files

After extraction, you should have:
- `diabetic_data.csv` - Main dataset (101,766 encounters, 55 features)
- `IDs_mapping.csv` - Mapping of admission/discharge codes to descriptions
- `dataset_diabetes_diabetesReadmis.names` - Feature descriptions

### Dataset Characteristics

- **Size:** 101,766 inpatient encounters
- **Hospitals:** 130 US hospitals
- **Time Period:** 1999-2008
- **Features:** 55 (demographics, diagnoses, medications, lab results, prior utilization)
- **Target Variable:** Readmission (<30 days, >30 days, No readmission)

### Key Features

**Demographics (4):**
- Age group, race, gender, weight

**Admission Details (5):**
- Admission type, admission source, discharge disposition, time in hospital, medical specialty

**Clinical Metrics (4):**
- Number of lab procedures, procedures, medications, diagnoses

**Diagnoses (3):**
- Primary, secondary, tertiary diagnoses (ICD-9 codes)

**Prior Utilization (3):**
- Outpatient visits, emergency visits, inpatient visits (prior year)

**Laboratory Results (2):**
- Glucose serum test result, HbA1c test result

**Medications (24):**
- Metformin, insulin, glyburide, and 21 other diabetes medications
- Each with dosage change indicator (Up/Down/Steady/No)

**Medication Management (2):**
- Change indicator, diabetes medication prescribed

### Target Variable Creation

For this project, we create a **binary target variable**:
- `readmit_binary = 1`: Readmission within 30 days
- `readmit_binary = 0`: No readmission OR readmission after 30 days

**Class Distribution:**
- Positive class (readmitted <30 days): 11.2%
- Negative class: 88.8%
- **Imbalance Ratio:** 7.2:1

### Missing Data

Some features have substantial missing values:
- **weight:** 97% missing → DROP
- **payer_code:** 52% missing → DROP
- **medical_specialty:** 47% missing → KEEP (use "Unknown" category)
- **race:** 2% missing → KEEP (impute with mode)
- **glucose_serum, A1C_result:** 82% "None" → KEEP (clinically meaningful - test not performed)

### Data Quality Issues

1. **Duplicate encounters:** Some patients have multiple encounters
   - **Solution:** Keep only first encounter per patient

2. **Invalid discharges:** Patients discharged to hospice or deceased
   - **Solution:** Remove these encounters (not readmission candidates)

3. **High-cardinality categoricals:** Medical specialty (73 unique values)
   - **Solution:** Use target encoding instead of one-hot encoding

### Expected Final Dataset

After preprocessing:
- **Encounters:** ~70,000 (from 101,766 original)
- **Features:** 55 original + 22 engineered = 77 total
- **Target:** Binary readmission (11.2% positive class)

### Citation

If you use this dataset, please cite:

```
Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., 
& Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: 
Analysis of 70,000 clinical database patient records. BioMed Research International, 
2014, 781670. https://doi.org/10.1155/2014/781670
```

### Dataset License

This dataset is publicly available from the UCI Machine Learning Repository for research purposes.

---

**Note:** The actual dataset file (`diabetic_data.csv`) is NOT included in this repository due to its size (20 MB). Please download it from the UCI repository using the instructions above.
