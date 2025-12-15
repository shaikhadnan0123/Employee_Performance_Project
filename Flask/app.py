from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load encoder and model
with open(os.path.join(BASE_DIR, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

with open(os.path.join(BASE_DIR, "model_xgb.pkl"), "rb") as f:
    model = pickle.load(f)

# Initialize Flask
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

    date = int(datetime.now().timestamp())

    df = pd.DataFrame([{
        'date': date,
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

    # Only transform, do NOT fit again
    df_encoded = encoder.transform(df)

    prediction = model.predict(df_encoded)[0]

    if prediction < 0.3:
        text = "The employee is averagely productive."
    elif prediction <= 0.8:
        text = "The employee is medium productive."
    else:
        text = "The employee is highly productive."

    return render_template('Submit.html', prediction_text=text)

if __name__ == "__main__":
    # Dynamic port for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
