import json
import numpy as np
import tensorflow as tf
import tensorflow_hub 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from keras import layers
import joblib

template = '{"autoFuel": [],"transitionFuel":[], "shift1Fuel":[], "shift2Fuel":[],"shift3Fuel":[],"shift4Fuel":[],"endgameFuel":[]}'
data1 = json.loads(template)
data_home = json.loads(template)
team_numbers = []
match_ids = []

SAVE_MODE = True
MODEL_FILENAME = "frc_score_predictions.keras"
SCALER_FILENAME = 'frc_scaler.pkl'
#random data
def prepare_data(data_path):
    x_data = []
    y_data = []
    with open(data_path, "r") as f:
        content = json.load(f)
    for teamnum, matches in content["root"].items():
        team_numbers.append(teamnum)
        for match_id, match_data in matches.items():
            score = [
                match_data.get('autoFuel', 0),
                match_data.get('transitionFuel', 0),
                match_data.get('shift1Fuel', 0),
                match_data.get('shift2Fuel', 0),
                match_data.get('shift3Fuel', 0),
                match_data.get('shift4Fuel', 0),
                match_data.get('endgameFuel', 0)
            ]  
            total_scores = sum(score)
            x_data.append(score)
            y_data.append(total_scores)
    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32)

model = tf.keras.Sequential([
    layers.Input(shape=(7,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
x, y, raw_json = prepare_data("m.json")
scaler = StandardScaler()
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=0)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=0)

x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

graph = model.fit(
    x_train_scaled, 
    y_train,
    batch_size=16,
    epochs=300,
    verbose=1,
    validation_data=(x_valid_scaled, y_valid)
)
#comparing to simulation



def plot(graph):
    plt.figure(figsize=(10, 6))
    plt.plot(graph.history['mae'], label='Training MAE')
    plt.plot(graph.history['val_mae'], label='Val MAE')
    plt.title('Model Learning')
    plt.xlabel('Epoch')
    plt.ylabel('Points Error')
    plt.legend()
    plt.grid(True)
    plt.show()
plot(graph)
model.save("frc_scoring_predictions.keras")
joblib.dump(scaler, "frc_scaler.pkl")
print(f"saved model to {MODEL_FILENAME} and scaler to {SCALER_FILENAME}")


