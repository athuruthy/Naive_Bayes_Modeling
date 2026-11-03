# Predicting Power Plant Fuel Category via Emission Fingerprints

## Overview
This project evaluates the extent to which a power plant's primary fossil fuel source can be identified solely through its annual emissions profile. Using 2007 plant data from the EPA's Emissions & Generation Resource Integrated Database (eGRID), we model the relationship between key air emissions and fuel categories using a Gaussian Naive Bayes classifier.

## Data & Features
- **Data Source:** [EPA Historical eGRID Data (2007)](https://www.epa.gov/egrid/historical-egrid-data) (`PLNT07` tab)
- **Target Variable:** `PLFUELCT` — Primary Fossil Fuel Category (`COAL`, `GAS`, `OIL`, `OFSL`)
- **Predictor Variables:**
  - `PLCO2AN`: Annual $CO_2$ emissions (tons)
  - `PLSO2AN`: Annual $SO_2$ emissions (tons)
  - `PLNOXAN`: Annual $NO_x$ emissions (tons)
  - `PLCH4AN`: Annual $CH_4$ emissions (tons)

## Methodology
1. **Data Preprocessing & Transformation:** Raw emissions span multiple orders of magnitude across fuel types, producing heavy right-skew that violates Gaussian assumptions. A `log1p` transformation was applied to all four predictor distributions.
2. **Train/Test Split:** 70% training set and 30% held-out test set.
3. **Modeling:** Supervised Gaussian Naive Bayes classification fit using R.

## Key Findings & Performance
- **Overall Accuracy:** 77.64% on the test set.
- **Coal Identification:** High precision and recall due to distinctive, elevated $SO_2$ emission profiles.
- **Class Imbalance & Limitations:** The classifier struggled significantly with underrepresented classes (e.g., Other Fossil / `OFSL` yielded a 0% recall rate). Severe representational bias in real-world generation portfolios causes the Naive Bayes prior and likelihood estimates to default to dominant categories (`GAS` and `COAL`) under overlapping emission signatures.

## Tech Stack
- **Language:** R
- **Key Packages:** `e1071` / `naivebayes`, `caret`, `ggplot2`, `dplyr`, `tidyr`, `pander`
