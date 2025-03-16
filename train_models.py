import pandas as pd
import numpy as np
import xgboost as xgb
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Load Data
file_path = "data_filtered.csv"
df = pd.read_csv(file_path)

# Convert time columns to datetime
df["time"] = pd.to_datetime(df["time"])
df["timeAtDestination"] = pd.to_datetime(df["timeAtDestination"])

# Feature Engineering
df["travel_duration"] = (df["timeAtDestination"] - df["time"]).dt.total_seconds() / 60 
df["hour"] = df["time"].dt.hour 
df["day_of_week"] = df["time"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

# Label Encoding for categorical variables
label_encoders = {}
categorical_columns = ["lineID", "stationID", "deviceID", "payType"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  

# Encode the target variable (destinationStation)
target_encoder = LabelEncoder()
df["destinationStation_encoded"] = target_encoder.fit_transform(df["destinationStation"])

# Define features and target variable
features = ["lineID", "stationID", "deviceID", "payType", "hour", "day_of_week", "is_weekend"]
target = "destinationStation_encoded"  # Use the encoded labels

X = df[features]
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multinomial Logit Model (MNL)
X_train_mnl = sm.add_constant(X_train)
mnl_model = sm.MNLogit(y_train, X_train_mnl)
mnl_result = mnl_model.fit()

# Save the MNL Model
with open("mnl_model.pkl", "wb") as f:
    pickle.dump(mnl_result, f)

print("Multinomial Logit Model trained & saved!")

# XGBoost Model (Now using the correctly encoded labels)
xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(target_encoder.classes_), eval_metric="mlogloss")
xgb_model.fit(X_train, y_train)

# Save XGBoost Model
xgb_model.save_model("xgb_model.json")

print("XGBoost Model trained & saved!")

# Save Label Encoders (Including target encoder)
label_encoders["destinationStation"] = target_encoder  # Save the target encoder
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("Label encoders saved!")
