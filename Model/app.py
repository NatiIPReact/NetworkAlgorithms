from flask import Flask, render_template, request, jsonify
import folium
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the trained model and scaler
model = tf.keras.models.load_model('bike_sharing_model.h5')
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Coordinates and radii for the zones in Chicago
centroids = [
    (41.8872954731507, -87.6364204090411),
    (41.8397556993617, -87.61534321212767),
    (41.8584000214634, -87.65814208195121),
    (41.94104849653059, -87.65579373867345),
    (41.90844586829786, -87.67549568553193)
]
radii = [
    1176.9973973066046,
    2920.5655895248883,
    1615.0571144046332,
    2110.732769808152,
    1588.9685603075486
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    proximity_value = data.get('proximity_value')
    median_income = data.get('median_income')

    # Validate input values
    if proximity_value is None or median_income is None:
        return jsonify({'error': 'Missing input values'}), 400

    try:
        proximity_value = float(proximity_value)
        median_income = float(median_income)
    except ValueError:
        return jsonify({'error': 'Invalid input values'}), 400

    if not (0 <= proximity_value <= 1):
        return jsonify({'error': 'Proximity value must be between 0 and 1'}), 400
    if median_income < 0:
        return jsonify({'error': 'Median income must be non-negative'}), 400

    # Prepare the sample input for prediction
    sample_features = pd.DataFrame({
        'proximity_value': [proximity_value],
        'SourceMedianIncome': [median_income]
    })

    # Scale the sample features
    scaled_sample_features = scaler.transform(sample_features)

    # Predict the community zone
    predicted_prob = model.predict(scaled_sample_features)
    predicted_class = np.argmax(predicted_prob, axis=1)[0]

    # Get coordinates and radius for the predicted zone
    coordinates = centroids[predicted_class]
    radius = radii[predicted_class]

    # Create a Folium map centered around the given coordinates
    m = folium.Map(location=[coordinates[0], coordinates[1]], zoom_start=14)

    # Add the centroid and radius circle to the map
    folium.Marker(
        location=[coordinates[0], coordinates[1]],
        popup='Predicted Zone Centroid',
        icon=folium.Icon(color='blue')
    ).add_to(m)

    folium.Circle(
        location=[coordinates[0], coordinates[1]],
        radius=radius,  # radius in meters
        color='blue',
        fill=True,
        fill_opacity=0.2
    ).add_to(m)

    # Render the map in HTML
    map_html = m._repr_html_()

    return jsonify({'map': map_html})

if __name__ == '__main__':
    app.run(debug=True)
