from flask import Flask, request, jsonify, render_template
import pandas as pd
import xgboost as xgb
import statsmodels.api as sm
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

app = Flask(__name__)

# Load Models
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgb_model.json")

with open("mnl_model.pkl", "rb") as f:
    mnl_model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

target_encoder = label_encoders["destinationStation"]

# **Route to Serve the Web Page**
@app.route("/")
def home():
    return render_template("index.html")

def plot_to_base64(fig):
    img = BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# **Generate Congestion Over Time (Line Chart)**
@app.route("/congestion_plot")
def congestion_plot():
    try:
        df = pd.read_csv("data_filtered.csv", parse_dates=["time"])
        df["hour"] = df["time"].dt.hour
        congestion_data = df.groupby("hour")["userID"].count().reset_index()
        
        # Normalize congestion levels
        congestion_data["congestion_level"] = (congestion_data["userID"] - congestion_data["userID"].min()) / \
                                              (congestion_data["userID"].max() - congestion_data["userID"].min())

        # **Create Plot**
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=congestion_data, x="hour", y="congestion_level", marker="o", color="red", ax=ax)
        ax.set_title("Metro Congestion Levels Over Time")
        ax.set_xlabel("Hour of the Day")
        ax.set_ylabel("Congestion Level (Normalized)")
        plt.grid()

        return jsonify({"image": plot_to_base64(fig)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# **Generate Peak vs Off-Peak Passenger Flow (Bar Chart)**
@app.route("/peak_off_peak_plot")
def peak_off_peak_plot():
    try:
        df = pd.read_csv("data_filtered.csv", parse_dates=["time"])
        df["hour"] = df["time"].dt.hour

        peak_hours = df[df["hour"].between(7, 10) | df["hour"].between(17, 20)]
        off_peak_hours = df[~df["hour"].between(7, 10) & ~df["hour"].between(17, 20)]
        peak_counts = peak_hours.groupby("stationID")["userID"].count().reset_index()
        off_peak_counts = off_peak_hours.groupby("stationID")["userID"].count().reset_index()

        # Fix: Create properly spaced bars for readability
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_width = 0.4
        stations = peak_counts["stationID"].unique()
        x = range(len(stations))

        peak_values = peak_counts.set_index("stationID")["userID"]
        off_peak_values = off_peak_counts.set_index("stationID")["userID"]

        ax.bar([i - bar_width / 2 for i in x], peak_values, width=bar_width, color="red", label="Peak Hours")
        ax.bar([i + bar_width / 2 for i in x], off_peak_values, width=bar_width, color="blue", label="Off-Peak Hours")

        ax.set_title("Passenger Flow: Peak vs Off-Peak")
        ax.set_xlabel("Station")
        ax.set_ylabel("Passenger Count")
        ax.set_xticks(x)
        ax.set_xticklabels(stations, rotation=45, ha="right")
        ax.legend()

        return jsonify({"image": plot_to_base64(fig)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# **Generate MNL OD Matrix (Heatmap)**
@app.route("/od_matrix_mnl")
def od_matrix_mnl():
    try:
        df = pd.read_csv("data_filtered.csv")
        od_matrix_mnl = df.groupby(["stationID", "destinationStation"]).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(od_matrix_mnl, cmap="coolwarm", ax=ax)
        ax.set_title("MNL - Origin-Destination Flow Heatmap")
        ax.set_xlabel("Destination Station")
        ax.set_ylabel("Origin Station")

        return jsonify({"image": plot_to_base64(fig)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# **Generate XGBoost OD Matrix (Heatmap)**
@app.route("/od_matrix_xgb")
def od_matrix_xgb():
    try:
        # Load dataset
        df = pd.read_csv("data_filtered.csv", parse_dates=["time"])

        # Ensure time is in datetime format with the correct precision
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

        with open("label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)

        target_encoder = label_encoders["destinationStation"]

        # Encode categorical variables
        def encode_value(column, value):
            value = str(value)
            if value in label_encoders[column].classes_:
                return label_encoders[column].transform([value])[0]
            else:
                return label_encoders[column].transform([label_encoders[column].classes_[0]])[0]

        df["lineID"] = df["lineID"].astype(str).apply(lambda x: encode_value("lineID", x))
        df["stationID"] = df["stationID"].astype(str).apply(lambda x: encode_value("stationID", x))
        df["deviceID"] = df["deviceID"].astype(str).apply(lambda x: encode_value("deviceID", x))
        df["payType"] = df["payType"].astype(str).apply(lambda x: encode_value("payType", x))

        # **Fix datetime issue**
        df["hour"] = df["time"].dt.hour
        df["day_of_week"] = df["time"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

        # Select features for XGBoost prediction
        X_features = df[["lineID", "stationID", "deviceID", "payType", "hour", "day_of_week", "is_weekend"]]

        # **Make XGBoost Predictions**
        df["xgb_pred_encoded"] = xgb_model.predict(X_features)
        df["xgb_pred"] = target_encoder.inverse_transform(df["xgb_pred_encoded"])

        # Generate OD Matrix using XGBoost-predicted destinations
        od_matrix_xgb = df.groupby(["stationID", "xgb_pred"]).size().unstack(fill_value=0)

        # **Plot the XGBoost OD Heatmap**
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(od_matrix_xgb, cmap="coolwarm", ax=ax)
        ax.set_title("XGBoost - Origin-Destination Flow Heatmap")
        ax.set_xlabel("Destination Station")
        ax.set_ylabel("Origin Station")

        return jsonify({"image": plot_to_base64(fig)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route("/congestion_alert", methods=["GET"])
def congestion_alert():
    try:
        # Load dataset
        df = pd.read_csv("data_filtered.csv", parse_dates=["time"])

        # Compute congestion per hour
        df["hour"] = df["time"].dt.hour
        congestion_data = df.groupby("hour")["userID"].count().reset_index()
        
        # Normalize congestion
        congestion_data["congestion_level"] = (congestion_data["userID"] - congestion_data["userID"].min()) / \
                                              (congestion_data["userID"].max() - congestion_data["userID"].min())

        # Define congestion threshold (e.g., 80% or above is high congestion)
        congestion_threshold = 0.8
        congestion_data["alert"] = congestion_data["congestion_level"].apply(lambda x: "High" if x >= congestion_threshold else "Normal")

        # Find the most congested hours
        peak_hours = congestion_data[congestion_data["alert"] == "High"]["hour"].tolist()

        response = {
            "peak_hours": peak_hours,
            "message": "High congestion expected during these hours."
        } if peak_hours else {
            "message": "No significant congestion detected."
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/passenger_flow", methods=["GET"])
def passenger_flow():
    try:
        # Load dataset
        df = pd.read_csv("data_filtered.csv", parse_dates=["time"])

        # Count number of passengers per station
        station_counts = df.groupby("stationID")["userID"].count().reset_index()
        station_counts = station_counts.rename(columns={"userID": "passenger_count"})

        # Find busiest stations
        top_stations = station_counts.sort_values(by="passenger_count", ascending=False).head(10)

        response = {
            "passenger_flow": station_counts.to_dict(orient="records"),
            "top_stations": top_stations.to_dict(orient="records")
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/alternative_routes", methods=["POST"])
def alternative_routes():
    try:
        # Parse JSON input
        data = request.json
        start_station = int(data["start_station"])
        destination_station = int(data["destination_station"])

        print(f"API Debug: Start={start_station}, Destination={destination_station}")

        # Load dataset
        df = pd.read_csv("data_filtered.csv")

        # Convert to datetime
        df["time"] = pd.to_datetime(df["time"])
        df["timeAtDestination"] = pd.to_datetime(df["timeAtDestination"])

        # Compute congestion per station
        station_congestion = df.groupby("stationID")["userID"].count().reset_index()
        station_congestion = station_congestion.rename(columns={"userID": "congestion_level"})

        print(f" Available Stations in Data: {df['stationID'].unique()}")

        # Find direct routes
        routes = df[(df["stationID"] == start_station) & (df["destinationStation"] == destination_station)]

        if routes.empty:
            print("No direct routes found.")
            return jsonify({"error": "No direct route found. Consider transfers."}), 400

        # Compute alternative routes
        alternative_routes = []
        for _, row in routes.iterrows():
            congestion = station_congestion.loc[station_congestion["stationID"] == row["stationID"], "congestion_level"].values[0]

            # Fix: Ensure datetime subtraction works
            travel_time = (row["timeAtDestination"] - row["time"]).total_seconds() // 60  # Travel time in minutes

            alternative_routes.append({
                "via": int(row["stationID"]),
                "travel_time": int(travel_time),  
                "congestion_level": int(congestion) 
            })

        print(f"Found {len(alternative_routes)} alternative routes.")

        # Sort routes by least congestion and travel time
        alternative_routes = sorted(alternative_routes, key=lambda x: (x["congestion_level"], x["travel_time"]))

        response = {
            "alternative_routes": alternative_routes[:3]  # Return top 3 alternative routes
        }

        return jsonify(response)

    except Exception as e:
        print(f"API Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route("/peak_fare", methods=["POST"])
def peak_fare():
    try:
        data = request.json
        base_fare = float(data["base_fare"])
        travel_time = datetime.strptime(data["travel_time"], "%H:%M").hour  

        # Define fare adjustments
        peak_hours = [7, 8, 9, 17, 18, 19]
        off_peak_hours = [10, 11, 12, 13, 14, 15, 16, 20, 21]

        if travel_time in peak_hours:
            adjusted_fare = round(base_fare * 1.2, 2)  # 20% increase
            fare_type = "Peak Hour Pricing"
        elif travel_time in off_peak_hours:
            adjusted_fare = round(base_fare * 0.8, 2)  # 20% discount
            fare_type = "Off-Peak Discount"
        else:
            adjusted_fare = base_fare
            fare_type = "Standard Fare"

        return jsonify({"adjusted_fare": adjusted_fare, "type": fare_type})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

@app.route("/station_fare_adjustment", methods=["POST"])
def station_fare_adjustment():
    try:
        data = request.json
        base_fare = data.get("base_fare")
        station_id = data.get("station_id")

        # Ensure required fields exist
        if base_fare is None or station_id is None:
            return jsonify({"error": "Missing base_fare or station_id"}), 400

        base_fare = float(base_fare)
        station_id = int(station_id)

        # Define station-specific pricing (Example: Distance-based adjustments)
        fare_adjustments = {
            range(1, 20): 1.0,  # No change
            range(20, 40): 1.1,  # 10% increase
            range(40, 60): 1.2,  # 20% increase
            range(60, 80): 1.3   # 30% increase
        }

        adjusted_fare = base_fare  # Default to base fare

        for station_range, multiplier in fare_adjustments.items():
            if station_id in station_range:
                adjusted_fare = round(base_fare * multiplier, 2)
                break

        return jsonify({"adjusted_fare": adjusted_fare})

    except Exception as e:
        return jsonify({"error": str(e)}), 400




# Payment Method Fare Discount
@app.route("/payment_method_fare", methods=["POST"])
def payment_method_fare():
    try:
        data = request.json
        base_fare = float(data["base_fare"])
        pay_type = str(data["pay_type"]).lower()  

        payment_discounts = {
            "metro_card": 0.9,   
            "digital_wallet": 0.9,   
            "credit_card": 0.95,  
            "debit_card": 0.95,   
            "cash": 1.0  
        }

        discount_factor = payment_discounts.get(pay_type, 1.0)  
        adjusted_fare = round(base_fare * discount_factor, 2)

        return jsonify({
            "payment_method": pay_type,
            "original_fare": base_fare,
            "adjusted_fare": adjusted_fare,
            "discount_applied": f"{(1 - discount_factor) * 100:.0f}%" if discount_factor < 1.0 else "No Discount"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# **API Route for Prediction**
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        lineID = str(data["lineID"])
        stationID = str(data["stationID"])
        deviceID = str(data["deviceID"])
        payType = str(data["payType"])
        time = datetime.strptime(data["time"], "%Y-%m-%d %H:%M:%S")

        hour = time.hour
        day_of_week = time.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

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

        # **XGBoost Prediction**
        predicted_station_encoded = int(xgb_model.predict(X_input)[0])
        predicted_station = target_encoder.inverse_transform([predicted_station_encoded])[0]

        # **MNL Model Factor Influence**
        X_input_mnl = sm.add_constant(X_input, has_constant='add')
        mnl_probs = mnl_model.predict(X_input_mnl)

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
