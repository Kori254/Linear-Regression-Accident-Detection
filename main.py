import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load the dataset
file_path = 'Road_dataset.csv'
data = pd.read_csv(file_path)

# Select relevant features and target variable
features = ['Time', 'Day_of_week', 'Age_band_of_driver', 'Type_of_vehicle', 
            'Driving_experience', 'Cause_of_accident']
target = 'Accident_severity'

X = data[features]
y = data[target]

# Encode the target variable (Accident_severity)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# One-Hot Encode the categorical features
column_transformer = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), features)
], remainder='passthrough')

X = column_transformer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'accident_severity_model.pkl')

# Example prediction
example_data = pd.DataFrame([['17:02:00', 'Monday', '18-30', 'Lorry (41?100Q)', 'Below 1yr', 'Changing lane to the left']],
                            columns=features)

# Apply the same column transformer
example_data_transformed = column_transformer.transform(example_data)

# Predict accident severity
prediction = model.predict(example_data_transformed)
predicted_severity = label_encoder.inverse_transform([int(round(prediction[0]))])
print("Predicted Accident Severity:", predicted_severity[0])
