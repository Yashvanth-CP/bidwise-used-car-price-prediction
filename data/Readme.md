# Indian Used Car Listings Dataset (57,480 Clean Records | Multi-Source)

This dataset was created to address a practical gap:  
**there is no large, clean, India-specific used-car dataset suitable for machine learning price prediction.**

Public datasets are typically US-centric, sparsely populated, or filled with missing values.  
To solve this, a multi-source data collection pipeline was built to extract **structured listing metadata** directly from major Indian automotive marketplaces.

The result is a **57,480-row, fully unique, zero-null, ML-ready dataset** covering a wide geographic and brand spectrum across India.

---

## 1. Dataset Summary

- **Total Records:** 57,480  
- **Geographic Coverage:** Pan-India (multi-city)  
- **Sources:** Multiple major Indian used-car platforms  
- **Duplicates:** Fully removed via cross-source deduplication  
- **Missing Values:** 0%  
- **Primary Use Case:**  
  - Price prediction (regression)  
  - Automotive market analytics  
  - Feature engineering research  
  - ML benchmarking on Indian market data

---

## 2. Files Included

This release contains **two aligned versions** of the same dataset:

### 1) `indian_used_cars_raw_57480.csv`
- Human-readable dataset
- Cleaned categorical labels
- Suitable for:
  - Exploratory Data Analysis (EDA)
  - Visualization
  - Manual feature engineering
  - Academic research
  - Business analytics

### 2) `indian_used_cars_numeric_57058.csv`
- Fully numeric, **model-ready** dataset
- Suitable for:
  - Linear regression
  - Random Forest, XGBoost
  - Neural networks
  - End-to-end ML pipelines

Both files have identical row order and alignment.

---

## 3. Final Feature Set (10 Core Attributes)

Each row represents one unique used-car listing with the following fields:

| Column | Description |
|--------|-------------|
| `price` | Listed selling price |
| `manufacturing_year` | Year of manufacture |
| `model` | Vehicle model |
| `km_driven` | Odometer reading (in kilometers) |
| `fuel_type` | Petrol / Diesel / CNG / Electric |
| `transmission_type` | Manual / Automatic |
| `brand` | Manufacturer brand |
| `city` | Listing city |
| `body_type` | Hatchback / Sedan / SUV / etc. |
| `number_of_owners` | Previous ownership count |

These features represent the **core determinants of resale value** used in real-world automotive valuation systems.

---

## 4. Data Cleaning & Normalization

The raw scraped data was systematically cleaned and standardized:

- Label normalization  
  - e.g., `"MINI"` → `"Mini"`  
  - `"four or more owners"` → `"More than 4 Owners"`
- Category consolidation across sources  
- Format unification for:
  - ownership labels  
  - fuel types  
  - transmission types  
  - city names  
- Strict removal of:
  - duplicate listings  
  - mirrored city listings  
  - repeated platform entries  
- Final dataset contains **only fully valid rows**

---

## 5. Numeric Encoding Strategy (ML-Ready File)

The numeric dataset was generated using the following encoding rules:

### • Ordinal Encoding (Price-Influenced Ranking)
Used where **relative value ordering is meaningful for prediction**, for example:
- Ordinality was based on average listing price of each brand.
- Brand (economy -> premium -> luxury segments)
- Ownership categories

> Note: This is a **modeling assumption**, not an inherent ordinal property, applied strictly for regression performance.

### • One-Hot Encoding (Nominal Variables)
Used for:
- Fuel type  
- Transmission type  
- Body type  

### • Existing Numeric Fields Preserved
- `price`
- `km_driven`
- `manufacturing_year`


---

## 6. Data Collection Overview (High-Level)

- Data was collected using a **custom Python scraping pipeline**.
- Structured listing metadata was extracted directly from **machine-readable sources (JSON-LD / API responses)**.
- A **breadth-first geographic strategy** was used to maximize coverage.
- Multi-source merging was performed.
- Cross-platform duplicates were removed using strict matching keys.

All processing steps were designed for:
- reproducibility  
- schema consistency  
- low noise  
- ML usability

---

## 7. Recommended Use Cases

- Used-car price prediction models  
- Brand-wise and city-wise valuation analysis  
- Mileage vs depreciation studies  
- Fuel-type price behavior analysis  
- Ownership impact on resale pricing  
- Automated valuation model (AVM) benchmarks  
- Feature importance and explainability research

---

## 8. Example Usage (Kaggle)

```python
import pandas as pd

df = pd.read_csv("/kaggle/input/indian-used-cars/indian_used_cars_raw_57480.csv")
df.head()

```

For modeling:
```python
X = pd.read_csv("/kaggle/input/indian-used-cars/indian_used_cars_numeric_57480.csv").to_numpy()
```

## 9. Data Ethics & Usage

- All information was derived from publicly available vehicle listing data.  

- No personal seller information was collected.  

- This dataset is shared strictly for research and educational purposes.  

- Original platforms retain ownership of individual listings.  

## 10. Citation

If you use this dataset in projects, publications, or notebooks, please cite as:  

> Shashi (2025). Indian Used Car Listings Dataset (57,480 Records).  

## 11. Versioning  
- v1.0 - Initial public release  
    - 57,480 clean, deduplicated records.
    - Raw + ML-ready numeric versions.  
    - Multi-source integration.  

## Final Note:
This dataset was built end-to-end:  
**collection -> normalization -> deduplication -> encoding -> ML-ready release.**  

A public dataset did not exist for the Indian market, it was created.  
