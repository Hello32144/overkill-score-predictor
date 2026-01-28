import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

model = tf.keras.models.load_model("frc_scoring_predictions.keras")
scaler = joblib.load('frc_scaler.pkl')

def get_team_data(team_id, data):
    team_matches = data['root'].get(str(team_id))
    match_list = []
    for m_id, m_data in team_matches.items():
        match_list.append([
            m_data.get('autoFuel', 0), m_data.get('transitionFuel', 0),
            m_data.get('shift1Fuel', 0), m_data.get('shift2Fuel', 0),
            m_data.get('shift3Fuel', 0), m_data.get('shift4Fuel', 0),
            m_data.get('endgameFuel', 0)
        ])
    return np.mean(match_list, axis=0).reshape(1, -1)

def run_comparison(team1, team2, data):
    t1_avg = get_team_data(team1, data)
    t2_avg = get_team_data(team2, data)
    
    if t1_avg is not None and t2_avg is not None:
        t1_scaled = scaler.transform(t1_avg)
        t2_scaled = scaler.transform(t2_avg)
        
        p1 = model.predict(t1_scaled, verbose=0)[0][0]
        p2 = model.predict(t2_scaled, verbose=0)[0][0]
        
        print(f"{team1} score:{p1}")
        print(f"{team2} score: {p2}")


run_comparison("3464", "254", "m.json")