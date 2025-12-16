from flask import Flask, render_template, request
import pandas as pd
import os
from datetime import datetime
import pickle

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_xgb.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")

# ---------------------------
# Load model & encoder
# ---------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/predict')
def predict():
    return render_template('Predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # ---- GET FORM VALUES ----
        quarter = int(request.form['quarter'])
        department = request.form['department']
        day = request.form['day']
        team = int(request.form['team'])
        targeted_productivity = float(request.form['targeted_productivity'])
        smv = float(request.form['smv'])
        wip = float(request.form['wip'])
        over_time = int(request.form['over_time'])
        incentive = int(request.form['incentive'])
        idle_time = float(request.form['idle_time'])
        idle_men = int(request.form['idle_men'])
        no_of_style_change = int(request.form['no_of_style_change'])
        no_of_workers = float(request.form['no_of_workers'])

        # ---- DATE (numeric as in training) ----
        date_value = datetime.today().toordinal()

        # ---- CREATE DATAFRAME ----
        df = pd.DataFrame([{
            'date': date_value,
            'quarter': quarter,
            'department': department,
            'day': day,
            'team': team,
            'targeted_productivity': targeted_productivity,
            'smv': smv,
            'wip': wip,
            'over_time': over_time,
            'incentive': incentive,
            'idle_time': idle_time,
            'idle_men': idle_men,
            'no_of_style_change': no_of_style_change,
            'no_of_workers': no_of_workers
        }])

        # ---- APPLY TRAINED ENCODER ----
        # If encoder was pickled after fitting
        if hasattr(encoder, "transform"):
            df_encoded = encoder.transform(df)
        else:
            df_encoded = encoder.fit_transform(df)  # fallback if transform not present

        # ---- PREDICTION ----
        prediction = model.predict(df_encoded)[0]

        # ---- RESULT LOGIC ----
        if prediction < 0.3:
            text = "The employee is averagely productive."
        elif prediction <= 0.8:
            text = "The employee is medium productive."
        else:
            text = "The employee is highly productive."

        return render_template('Submit.html', prediction_text=text)

    except Exception as e:
        return f"Error occurred: {e}"

# ---------------------------
# Run Flask
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # for Render
    app.run(host="0.0.0.0", port=port, debug=True)
