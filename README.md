# TUKAN | Mexican Government Bond Yield Curve Analysis and Forecasting

### Financial Data Analytics, Principal Component Analysis, and Time Series Forecasting with Python

**Project Type:** Financial Data Science Research  
**Field:** Data Science | Financial Analytics | Machine Learning | Econometrics  
**Author:** Data Engineer Luis Barahona  
**Environment:** Google Colab | Jupyter Notebook  
**Study Period:** October 2006 – September 2026

---

## 1. Project Overview

This project analyzes the historical behavior and predictability of Mexico's nominal government bond yield curve using monthly auction rate data from October 2006 to September 2026.

The study focuses on nine debt maturities, including short-term CETES and medium- to long-term Bonos M.

The research combines exploratory data analysis, Principal Component Analysis (PCA), statistical forecasting, and machine learning to investigate the structure of interest rates and forecast their behavior at 3- and 12-month horizons.

Four approaches are evaluated:

- Persistence baseline
- Damped Holt exponential smoothing
- Ridge Regression
- PCA + Ridge Regression

The analysis applies expanding-window temporal validation to compare forecasting performance without incorporating future observations into model training.

## 2. Research Objectives

The project addresses three main research questions:

1. Do the nine Mexican government debt maturities share a common underlying statistical structure?
2. Can their interest rates be forecast 3 and 12 months ahead?
3. Does PCA-based dimensionality reduction improve predictive performance?

Additional objectives include:

- Analyze historical yield curve movements.
- Identify relationships between maturities.
- Evaluate data completeness and missing observations.
- Extract common yield curve factors through PCA.
- Compare traditional and machine learning forecasting techniques.
- Measure forecasting errors across different maturities.
- Assess prediction stability over time.
- Evaluate uncertainty through temporal block bootstrap analysis.

---

## 3. Dataset Description

The project uses:

**Dataset:** `banxico_auction_rates_monthly.csv`

**Period:** October 2006 – September 2026

**Source context:** Banco de México auction rate information.

### Financial Instruments

| Instrument | Maturity | Category |
|:---|:---|:---|
| CETES | 28 days | Short-term |
| CETES | 91 days | Short-term |
| CETES | 182 days | Short-term |
| CETES | 364 days | Short-term |
| Bonos M | 3 years | Medium-term |
| Bonos M | 5 years | Medium-term |
| Bonos M | 10 years | Long-term |
| Bonos M | 20 years | Long-term |
| Bonos M | 30 years | Long-term |

The dataset contains monthly average auction rates.

These observations should not be interpreted as simultaneous market yield quotations.

### Dataset Access

[Download the CSV dataset from Google Drive](https://drive.google.com/file/d/1Ut2jTHjY3xcQY6rUSS0BLdXsDmTtlzMh/view)

---

## 4. Methodology

### 4.1 Data Loading and Quality Assessment

The analysis begins by importing the CSV dataset using Pandas.

The notebook examines:

- Date coverage
- Available financial instruments
- Data types and numerical units
- Duplicate records
- Missing observations
- Monthly data availability
- Consecutive periods without observations

The original observations are preserved for visualization and evaluation.

### 4.2 Yield Curve Visualization

Historical interest rate movements are visualized across approximately 20 years.

The analysis includes:

- Historical interest rate charts
- Data availability heatmaps
- Yield curve comparisons across selected months
- Maturity-specific statistical analysis

These visualizations help identify changes in the level, slope, and shape of the Mexican yield curve.

### 4.3 Missing Data Treatment

Missing observations are handled using forward filling within the information available at each forecast origin.

The notebook also tracks the age of the last available observation for each instrument.

This approach avoids using future information when constructing forecasting inputs.

Missing target observations are excluded from error calculations rather than replaced with artificial values.

### 4.4 Principal Component Analysis

PCA is applied to interest rate levels and monthly changes to investigate common statistical factors.

The analysis examines:

- Explained variance ratios
- Cumulative explained variance
- Component loadings
- Reconstruction errors
- Differences between yield levels and monthly changes
- Sensitivity to standardization and missing data
- Structural stability across two decades

The first components are interpreted in terms of common yield curve movements, including level, slope, and curvature.

### 4.5 Forecasting Models

Four forecasting approaches are evaluated.

#### Model 1 — Persistence Baseline

The latest available rate is used as the future forecast.

This provides a benchmark for measuring whether more complex models improve predictive accuracy.

#### Model 2 — Damped Holt

Damped Holt exponential smoothing models the rate level and trend while reducing the influence of long-term trend extrapolation.

Damping parameters are evaluated using historical validation data.

#### Model 3 — Ridge Regression

Ridge Regression uses lagged yield curve observations to estimate future interest rates.

Regularization is applied to control coefficient magnitude and reduce sensitivity to correlated predictors.

#### Model 4 — PCA + Ridge Regression

PCA transforms the interest rate information into a reduced set of components.

These components are subsequently used as predictive features in Ridge Regression.

The objective is to evaluate whether dimensionality reduction preserves useful forecasting information.

---

## 5. Temporal Validation

The study uses chronological evaluation rather than random train-test splitting.

### Experimental Periods

| Stage | Period |
|:---|:---|
| Initial training history | Through September 2016 |
| Validation forecast origins | October 2016 – September 2019 |
| Final test forecast origins | October 2020 – September 2025 |
| Forecast horizons | 3 and 12 months |

The final test contains 60 forecast origins for each horizon.

At each origin, forecasting models use only information available up to that date.

Hyperparameter selection is performed using validation data, while the final test is reserved for evaluating generalization.

### Evaluation Metrics

Two main metrics are calculated:

**Mean Absolute Error (MAE)**

Measures the average absolute difference between predictions and observed interest rates.

**Root Mean Squared Error (RMSE)**

Measures prediction error while placing greater weight on larger deviations.

Errors are expressed in basis points (bp).

The project also evaluates:

- Performance by financial instrument
- Performance by forecasting horizon
- Improvements relative to the persistence baseline
- Temporal stability
- Exploratory uncertainty intervals

---

## 6. Experimental Results

### Final Test Performance

The following results are reported in the notebook's final test.

| Model | MAE 3M (bp) | RMSE 3M (bp) | MAE 12M (bp) | RMSE 12M (bp) |
|:---|---:|---:|---:|---:|
| Persistence | 56.93 | 69.50 | 158.68 | 204.67 |
| Damped Holt | 45.92 | **56.84** | **136.48** | **173.22** |
| Ridge | **44.72** | 57.58 | 170.07 | 195.44 |
| PCA + Ridge | 46.33 | 58.98 | 169.58 | 202.22 |

### Main Finding

**Damped Holt achieved the lowest RMSE at both forecasting horizons.**

Compared with the persistence baseline, Holt reduced RMSE by:

- **18.22% at 3 months**
- **15.37% at 12 months**

Ridge achieved the lowest 3-month MAE, while Holt produced the lowest 12-month MAE.

These results demonstrate that model rankings depend on the evaluation metric and forecasting horizon.

### Forecasting Performance

The study found measurable improvements over the persistence baseline.

However, forecasting accuracy decreased at the longer horizon, and improvements were not uniform across all maturities.

Performance gains were particularly noticeable among some short-term CETES instruments.

---

## 7. PCA Findings

The analysis identified a strong common structure among the nine interest rate maturities.

### Explained Variance

| Period | Variance Explained by First Two Components |
|:---|---:|
| First decade | 97.78% |
| Second decade | 98.34% |

The results suggest that a small number of components can represent much of the historical variation in yield levels.

The first component is associated with broad level movements, while subsequent components capture differences between short- and long-term maturities.

For monthly rate changes, the concentration is weaker.

The first three components explained approximately:

- 78.17% during the first decade
- 76.43% during the second decade

### Does PCA Improve Forecasting?

The final test did not provide clear evidence that PCA improved Ridge forecasting performance.

Compared with Ridge without PCA, PCA + Ridge produced a higher RMSE:

- 2.43% higher at 3 months
- 3.47% higher at 12 months

Exploratory uncertainty intervals for the differences included zero.

Therefore, the experiment does not establish that PCA consistently improves or worsens forecasting performance.

**Key insight:** A compact representation of historical data does not necessarily preserve all information relevant to future predictions.

---

## 8. Uncertainty and Stability Analysis

The notebook includes additional evaluation procedures.

### Temporal Block Bootstrap

A paired temporal block bootstrap is used to explore uncertainty in differences between forecasting models.

The analysis uses 2,000 bootstrap replications and examines alternative temporal block sizes.

These intervals provide exploratory evidence regarding forecast error differences.

### Stability Across Test Periods

The final test is divided into two equal periods:

- October 2020 – March 2023
- April 2023 – September 2025

The objective is to investigate whether forecasting improvements remain stable across different market periods.

Holt's improvement over the persistence baseline was observed in both halves of the final test.

---

## 9. Technologies Used

| Technology | Purpose |
|:---|:---|
| Python | Programming language |
| Pandas | Financial data processing |
| NumPy | Numerical analysis |
| Matplotlib | Financial data visualization |
| Scikit-learn | PCA, scaling, and Ridge Regression |
| Statsmodels | Holt exponential smoothing |
| Google Colab | Cloud notebook environment |
| CSV | Dataset storage and exchange |
| JSON | Experiment metadata and documentation |
| SHA-256 | Export integrity verification |

---

## 10. How to Run

### Google Colab

Open the notebook:

[Run TUKAN in Google Colab](https://colab.research.google.com/drive/19hOIHhRVxtImIsMlg3JhkQZFGCr-xjLB)

### Instructions

1. Open the notebook in Google Colab.
2. Download the `banxico_auction_rates_monthly.csv` dataset.
3. Run the first code cell.
4. Upload the CSV file when prompted.
5. Execute the remaining cells sequentially.
6. Review the historical analysis, PCA, forecasting results, and model comparisons.
7. Export the generated tables and research outputs if required.

### Local Python Environment

Install the core dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels ipython jupyter
```

The original notebook uses Google Colab-specific file upload and export procedures, which require adaptation when running locally.

---

## 11. Research Outputs

The notebook includes procedures for exporting research tables and experiment documentation.

Generated outputs include:

- Data quality assessments
- Monthly data availability summaries
- PCA explained variance and reconstruction results
- Model configuration comparisons
- Validation metrics
- Final test predictions
- Forecast accuracy by maturity
- Global forecasting performance
- Baseline improvement comparisons
- Bootstrap uncertainty analysis
- Temporal stability results
- Experiment metadata and integrity information

These outputs support reproducibility and further examination of the results.

---

## 12. Limitations

The research includes several important limitations:

- Monthly auction averages are not simultaneous market quotations.
- The dataset does not independently establish historical publication dates or revisions.
- Forward filling may affect the statistical properties of interest rate changes.
- Some months have missing observations for specific maturities.
- Forecasting performance varies by maturity and horizon.
- Bootstrap confidence intervals are exploratory.
- The uncertainty analysis does not include refitting models in each bootstrap replication.
- Predictive relationships do not establish economic causality.
- The evaluated models should not be interpreted as investment recommendations.

---

## 13. Future Development

Potential extensions include:

- Incorporating macroeconomic indicators and inflation expectations.
- Evaluating additional forecasting architectures.
- Exploring dynamic factor models.
- Expanding model stability analysis.
- Testing different validation periods.
- Evaluating additional regularization techniques.
- Developing interactive dashboards.
- Automating data ingestion and research reporting.

---

## 14. Project Resources

### Google Colab Notebook

[Open TUKAN Research Notebook](https://colab.research.google.com/drive/19hOIHhRVxtImIsMlg3JhkQZFGCr-xjLB)

### Dataset

[Download the CSV Dataset](https://drive.google.com/file/d/1Ut2jTHjY3xcQY6rUSS0BLdXsDmTtlzMh/view)

### Tukan Data Platform

[Tukan — Mexican Data Platform](https://tukanmx.com/)

Tukan is an external platform focused on economic and market data for Mexico. The link is provided as a related data resource and does not imply official affiliation, sponsorship, or validation of this independent research.

### Banco de México

[Bank of Mexico](https://www.banxico.org.mx/)

---

## 15. Conclusion

This project demonstrates how financial data analysis, statistical methods, and machine learning can be combined to study the Mexican government bond yield curve.

The results identify a strong common structure among interest rate maturities and demonstrate that relatively simple forecasting approaches can outperform a persistence benchmark.

Damped Holt achieved the lowest final-test RMSE at both evaluated horizons, while PCA-based dimensionality reduction did not provide a consistent predictive advantage.

The research highlights the importance of chronological validation, transparent benchmarking, careful treatment of missing data, and uncertainty analysis when developing financial forecasting models.

---

## Author

**Data Engineer Luis Barahona**

Mexican Data Engineer and 2026 graduate of Universidad del Caribe, passionate about technology, financial data analytics, Data Science, Machine Learning, and continuous learning.

---

## Contact

| Platform | Contact |
|:---|:---|
| Email | [190300395@ucaribe.edu.mx](mailto:190300395@ucaribe.edu.mx) |
| GitHub | [github.com/LuisBarahonaMX](https://github.com/LuisBarahonaMX) |
| LinkedIn | [linkedin.com/in/luis-barahona-mx](https://www.linkedin.com/in/luis-barahona-mx/) |
| Instagram | [@luisbarahonamx](https://www.instagram.com/luisbarahonamx/) |
| X | [@luisbarahonaMX](https://x.com/luisbarahonaMX) |
| Website | [SIKO AI Platform](https://sikoai.web.app/) |
