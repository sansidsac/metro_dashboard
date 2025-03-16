from flask import Flask, request, jsonify
import pandas as pd
import xgboost as xgb
import statsmodels.api as sm
import pickle
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Load XGBoost Model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgb_model.json")

# Load MNL Model
with open("mnl_model.pkl", "rb") as f:
    mnl_model = pickle.load(f)

# Load Label Encoders
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Load Target Label Encoder for XGBoost Output
target_encoder = label_encoders["destinationStation"]

# API Route for Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON Input
        data = request.json

        # Extract Features
        lineID = str(data["lineID"])
        stationID = str(data["stationID"])
        deviceID = str(data["deviceID"])
        payType = str(data["payType"])
        time = datetime.strptime(data["time"], "%Y-%m-%d %H:%M:%S")

        # Feature Engineering
        hour = time.hour
        day_of_week = time.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        # Encode Categorical Variables
        def encode_value(column, value):
            value = str(value)
            if value in label_encoders[column].classes_:
                return label_encoders[column].transform([value])[0]
            else:
                print(f"Warning: Unseen value '{value}' for {column}. Assigning default.")
                return label_encoders[column].transform([label_encoders[column].classes_[0]])[0]

        features = {
            "lineID": encode_value("lineID", lineID),
            "stationID": encode_value("stationID", stationID),
            "deviceID": encode_value("deviceID", deviceID),
            "payType": encode_value("payType", payType),
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend
        }

        X_input = pd.DataFrame([features])

        # Debug: Print input features
        print("Input Features for Prediction:")
        print(X_input)

        # Predict Destination Station (XGBoost)
        predicted_station_encoded = int(xgb_model.predict(X_input)[0])
        predicted_station = target_encoder.inverse_transform([predicted_station_encoded])[0]

        # Debug: Print XGBoost prediction
        print(f"Predicted Destination (Encoded): {predicted_station_encoded}")
        print(f"Predicted Destination (Decoded): {predicted_station}")

        # Ensure Correct Shape for MNL Prediction
        X_input_mnl = sm.add_constant(X_input, has_constant='add')
        mnl_probs = mnl_model.predict(X_input_mnl)

        # Check MNL Prediction Output
        print("MNL Probabilities Shape:", mnl_probs.shape)

        # **Find Most Influential Feature**
        max_prob_index = int(np.argmax(mnl_probs.values, axis=1)[0]) 
        predicted_class_prob = float(mnl_probs.iloc[0, max_prob_index])

        response = {
            "predicted_destination": int(predicted_station),
            "predicted_class_probability": float(predicted_class_prob)
        }

        return jsonify(response)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
