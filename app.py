from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime

# IMPORTANT: import class BEFORE loading pickle
from encoder import MultiColumnLabelEncoder

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model & encoder (DO NOT CREATE THEM HERE)
model = pickle.load(open(os.path.join(BASE_DIR, "model_xgb.pkl"), "rb"))
encoder = pickle.load(open(os.path.join(BASE_DIR, "encoder.pkl"), "rb"))

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

    # ---- GET FORM VALUES ----
    quarter = int(request.form['quarter'])
    department = request.form['department']   # STRING
    day = request.form['day']                 # STRING
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

    # ---- CREATE DATE (MODEL EXPECTS IT) ----
    date = int(datetime.now().timestamp())

    # ---- CREATE DATAFRAME (EXACT TRAINING COLUMNS) ----
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

    # ---- APPLY SAME ENCODER USED IN TRAINING ----
    df_encoded = encoder.fit_transform(df)

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

if __name__ == "__main__":

    app.run(debug=True)

