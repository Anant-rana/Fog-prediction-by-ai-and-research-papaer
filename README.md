# Fog Prediction Using Machine Learning

**AI-Powered Fog Detection Based on Dew Point, Temperature & Visibility**



## Problem Statement

Fog is a dangerous weather phenomenon that drastically reduces visibility, leading to serious risks for road transportation, aviation, and daily activities. Traditional meteorological models often struggle to predict fog accurately because it forms due to subtle, rapidly changing interactions between temperature, dew point, humidity, and local conditions.

In particular, **timely and localized fog prediction** remains challenging, especially in microclimates. This limitation increases the risk of multi-vehicle collisions, flight delays, and public safety incidents. A recent example is the pile-up of around 50 cars on the Toronto-Montreal highway due to dense fog.

There is a clear need for a data-driven, AI-based solution that can analyze complex weather patterns and provide reliable, early fog predictions.

## Solution

We developed a complete **machine learning pipeline** to predict fog (1 = Fog, 0 = No Fog) using historical weather data. The model focuses on key indicators like **Dew Point**, **Temperature**, **Visibility**, and the engineered **Temperature-Dew Point Spread**.

The system:
- Cleans and preprocesses raw weather data
- Creates meaningful features for fog formation
- Trains a robust classifier
- Delivers highly accurate predictions

This AI approach outperforms simple rule-based thresholds and offers potential for real-time integration into weather alert systems.

## Objectives

1. Analyze the relationship between dew point, temperature spread, and visibility for fog formation.
2. Build an end-to-end AI model for fog prediction.
3. Preprocess weather data with proper cleaning and feature engineering.
4. Train and evaluate machine learning models (Random Forest).
5. Demonstrate real-world benefits for transportation safety.

## Dataset Description

- **Filename**: `weather.csv`
- **Columns**: Temperature, DewPoint, Visibility, Humidity, WindSpeed
- **Size**: ~1000 records (realistic synthetic weather data)
- **Target**: Fog (binary) – defined as Visibility < 1 km

## Data Cleaning & Transformation

We performed thorough data preprocessing to ensure high-quality input for the model:

- Removed duplicate rows
- Handled missing values using **forward fill** (`ffill`) followed by mean imputation for any remaining NaNs
- Created a new critical feature:  
  **`Temp_Dew_Spread`** = Temperature - DewPoint  
  *(Smaller spread indicates higher likelihood of fog formation)*
- Labeled the target variable:  
  `Fog = 1` if Visibility < 1 km, else `0`
- Selected relevant features for modeling
- Applied **StandardScaler** for feature normalization

These steps significantly improved data quality and helped the model learn meaningful patterns.

## Methodology & Model

### Feature Engineering
- **Temp_Dew_Spread**: One of the most important indicators for fog prediction

### Train-Test Split
- 80% training, 20% testing
- Random state = 42 for reproducibility

### Machine Learning Model
- **Algorithm**: `RandomForestClassifier(n_estimators=200, random_state=42)`
- **Why Random Forest?** Robust, handles non-linear relationships well, resistant to overfitting, and provides good interpretability.
- Scaled features using `StandardScaler`

### Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Classification Report

**Model Performance**: Achieved **Accuracy = 1.0** on the test set with perfect precision and recall for both classes.

## Results

### Place 1: Model Evaluation Screenshots
<img width="1376" height="593" alt="fog_prediction 1" src="https://github.com/user-attachments/assets/03077ff1-3f63-4aaf-9210-0edeeb4fd6de" />

**Key Results:**
- Accuracy: **1.0**
- Precision / Recall / F1-score: **1.00** for both Fog and No-Fog classes
- The model successfully learned the patterns between low visibility, dew point proximity, and fog occurrence.

### Place 2: Feature Distribution & Box Plot Analysis
<img width="930" height="632" alt="fog_prediction 2" src="https://github.com/user-attachments/assets/34901b70-0d77-4210-8da1-7117e67d145f" />


These visualizations help understand the distribution of key variables and how they differ between fog and no-fog conditions.

## Example Prediction

```python
# Sample Output from the notebook:
Fog Prediction (1=Fog, 0=No Fog): 1
