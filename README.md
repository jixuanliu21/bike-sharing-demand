# Bike Sharing Demand Prediction

## Overview

This project develops a machine learning workflow to predict hourly bike-sharing demand using historical rental records, weather conditions, and calendar features.

The objective is to explore demand patterns, compare predictive models, and evaluate their ability to generalise to later time periods.

## Dataset

**Source:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

- 17,379 hourly observations
- Period: January 2011 – December 2012
- Target variable: `cnt` (total hourly bike rentals)
- Features: hour, weekday, season, working-day status, weather conditions, temperature, humidity, and windspeed

The variables `casual` and `registered` were excluded from prediction features to prevent target leakage, as their sum equals `cnt`.

## Methodology

1. **Data Exploration:** Examined hourly demand, working-day patterns, and weather-related differences.
2. **Feature Engineering:** Selected temporal and environmental features.
3. **Data Splitting:** Used chronological training, validation, and test partitions to reduce temporal leakage.
4. **Model Development:** Compared a historical-average baseline, linear regression, and random forest regression.
5. **Model Evaluation:** Evaluated performance using MAE, RMSE, and R².

## Results

| Model | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| Historical Average | 124.63 | 166.63 | 0.409 |
| Linear Regression | 108.39 | 136.31 | 0.604 |
| Random Forest | **49.80** | **74.21** | **0.883** |

### Final Random Forest Test Performance

| Metric | Value |
|---|---:|
| MAE | **44.94** |
| RMSE | **69.81** |
| R² | **0.900** |

The final Random Forest model was retrained using the full training partition before evaluation on the held-out test set.

## Key findings

- Hour of day was the most influential feature according to the Random Forest's impurity-based feature importance.
- Working-day and non-working-day demand followed distinct hourly patterns.
- Random Forest substantially outperformed the historical-average baseline and linear regression on the validation set.
- Some high-demand periods were underestimated, indicating scope for further improvement.

## Technologies used

Python, pandas, NumPy, Matplotlib, scikit-learn, Jupyter Notebook.

## Structure

- `data/`: Dataset and evaluation results
- `figures/`: Exploratory and model evaluation visualisations
- `notebooks/`: Data analysis and machine learning experiments
- `requirements.txt`: Python dependencies

## Running

1. Create and activate a Python virtual environment.
2. Install dependencies using `pip install -r requirements.txt`.
3. Download the UCI Bike Sharing Dataset and place `hour.csv` inside `data/`.
4. Open `notebooks/01_bike_demand.ipynb` and run the cells sequentially.

## Limitations and Future Work

- The model was evaluated on a single dataset and chronological holdout period.
- Random Forest has limited ability to extrapolate beyond previously observed demand levels.
- Future work could explore time-series cross-validation, permutation importance, additional temporal features, and demand forecasting for resource allocation.