from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pickle

# ---------------------------
# MultiColumnLabelEncoder
# ---------------------------
class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            self.encoders[col] = {
                val: idx for idx, val in enumerate(sorted(X[col].unique()))
            }
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, mapping in self.encoders.items():
            X_copy[col] = X_copy[col].map(mapping)
        return X_copy

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_xgb.pkl")

# ---------------------------
# Load model
# ---------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Encoder (categorical only)
# ---------------------------
encoder = MultiColumnLabelEncoder(columns=["department", "day"])

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("Home.html")


@app.route("/about")
def about():
    return render_template("About.html")


@app.route("/predict")
def predict():
    return render_template("Predict.html")


@app.route("/submit", methods=["POST"])
def submit():
    try:
        # ---------------------------
        # Get form values
        # ---------------------------
        quarter = int(request.form["quarter"])
        department = request.form["department"]
        day = request.form["day"]
        team = int(request.form["team"])
        targeted_productivity = float(request.form["targeted_productivity"])
        smv = float(request.form["smv"])
        wip = float(request.form["wip"])
        over_time = int(request.form["over_time"])
        incentive = int(request.form["incentive"])
        idle_time = float(request.form["idle_time"])
        idle_men = int(request.form["idle_men"])
        no_of_style_change = int(request.form["no_of_style_change"])
        no_of_workers = float(request.form["no_of_workers"])

        # ---------------------------
        # REQUIRED date feature
        # ---------------------------
        date_value = datetime.now().toordinal()

        # ---------------------------
        # Create DataFrame (ORDER MATTERS)
        # ---------------------------
        df = pd.DataFrame([{
            "date": date_value,
            "quarter": quarter,
            "department": department,
            "day": day,
            "team": team,
            "targeted_productivity": targeted_productivity,
            "smv": smv,
            "wip": wip,
            "over_time": over_time,
            "incentive": incentive,
            "idle_time": idle_time,
            "idle_men": idle_men,
            "no_of_style_change": no_of_style_change,
            "no_of_workers": no_of_workers
        }])

        # ---------------------------
        # Encode categorical columns ONLY
        # ---------------------------
        df_encoded = df.copy()
        df_encoded[["department", "day"]] = encoder.fit_transform(
            df[["department", "day"]]
        )[["department", "day"]]

        # ---------------------------
        # Prediction
        # ---------------------------
        prediction = float(model.predict(df_encoded)[0])

        # ---------------------------
        # Result message
        # ---------------------------
        if prediction < 0.3:
            text = "The employee is averagely productive."
        elif prediction <= 0.8:
            text = "The employee is medium productive."
        else:
            text = "The employee is highly productive."

        return render_template(
            "Submit.html",
            prediction_text=text,
            prediction_value=round(prediction, 4)
        )

    except Exception as e:
        return render_template(
            "Submit.html",
            prediction_text="Error occurred",
            error_message=str(e)
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        # ---------------------------
        # Get JSON values
        # ---------------------------
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"success": False, "error_message": "No JSON input received"}), 400

        quarter = int(data.get("quarter", 1))
        department = data.get("department", "sweing")
        day = data.get("day", "Monday")
        team = int(data.get("team", 1))
        targeted_productivity = float(data.get("targeted_productivity", 0.8))
        smv = float(data.get("smv", 0.0))
        wip = float(data.get("wip", 0.0))
        over_time = int(data.get("over_time", 0))
        incentive = int(data.get("incentive", 0))
        idle_time = float(data.get("idle_time", 0.0))
        idle_men = int(data.get("idle_men", 0))
        no_of_style_change = int(data.get("no_of_style_change", 0))
        no_of_workers = float(data.get("no_of_workers", 0.0))

        # ---------------------------
        # REQUIRED date feature
        # ---------------------------
        date_value = datetime.now().toordinal()

        # ---------------------------
        # Create DataFrame (ORDER MATTERS)
        # ---------------------------
        df = pd.DataFrame([{
            "date": date_value,
            "quarter": quarter,
            "department": department,
            "day": day,
            "team": team,
            "targeted_productivity": targeted_productivity,
            "smv": smv,
            "wip": wip,
            "over_time": over_time,
            "incentive": incentive,
            "idle_time": idle_time,
            "idle_men": idle_men,
            "no_of_style_change": no_of_style_change,
            "no_of_workers": no_of_workers
        }])

        # ---------------------------
        # Encode categorical columns ONLY
        # ---------------------------
        df_encoded = df.copy()
        df_encoded[["department", "day"]] = encoder.fit_transform(
            df[["department", "day"]]
        )[["department", "day"]]

        # ---------------------------
        # Prediction
        # ---------------------------
        prediction = float(model.predict(df_encoded)[0])

        # ---------------------------
        # Result message
        # ---------------------------
        if prediction < 0.3:
            text = "The employee is averagely productive."
            level = "average"
        elif prediction <= 0.8:
            text = "The employee is medium productive."
            level = "medium"
        else:
            text = "The employee is highly productive."
            level = "high"

        return jsonify({
            "success": True,
            "prediction_value": round(prediction, 4),
            "prediction_text": text,
            "prediction_level": level
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error_message": str(e)
        }), 500


# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
