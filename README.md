# Linear-Regression-Accident-Detection
# Road Accident Severity Prediction

This project aims to analyze and predict the severity of road accidents using a linear regression model. The model is built using a dataset containing various features related to road accidents, such as time, day of the week, driver's age, vehicle type, driving experience, and the cause of the accident. The purpose of this project is to help improve traffic safety analysis and prevention, especially in underdeveloped countries.

## Dataset

The dataset used for this project includes the following columns:

- **Time**: The time of the accident.
- **Day_of_week**: The day of the week when the accident occurred.
- **Age_band_of_driver**: The age group of the driver.
- **Type_of_vehicle**: The type of vehicle involved in the accident.
- **Driving_experience**: The driving experience of the driver.
- **Cause_of_accident**: The primary cause of the accident.
- **Accident_severity**: The severity of the accident (target variable), categorized as 'Slight Injury', 'Serious Injury', etc.

## Project Steps

1. **Data Preprocessing**
   - The dataset is cleaned, and irrelevant columns are removed.
   - Categorical features are encoded using One-Hot Encoding.
   - The target variable (`Accident_severity`) is encoded with `LabelEncoder` to convert it into numerical format.

2. **Model Training**
   - The dataset is split into training and testing sets.
   - A linear regression model is trained using the training data.

3. **Prediction**
   - The trained model is used to predict accident severity for new data.
   - The prediction output is transformed back into its original categories for easier interpretation.

## Files

- `Road_dataset.csv`: The dataset containing road accident information.
- `accident_severity_model.pkl`: The trained linear regression model saved for future use.
- `accident_severity_prediction.py`: Python script for preprocessing, training the model, and making predictions.

## Requirements

- Python 3.x
- Required libraries: `pandas`, `scikit-learn`, `joblib`

You can install the required packages using:
```bash
pip install pandas scikit-learn joblib
