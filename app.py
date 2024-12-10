from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from threading import Thread
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load the model and preprocessing objects
try:
    with open("preprocessing_objects.pkl", "rb") as f:
        objects = pickle.load(f)
    encoder = objects['encoder']
    skills_binarizer = objects['skills_binarizer']
    hobbies_binarizer = objects['hobbies_binarizer']
    scaler = objects['scaler']
    pca = objects['pca']

    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    logger.info("Models and preprocessing objects loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models or preprocessing objects: {e}")
    raise

# Function to safely transform skills and hobbies
def safe_transform(binarizer, data, binarizer_type):
    try:
        binarizer.classes_ = np.unique(np.concatenate([binarizer.classes_, data]))
        return binarizer.transform([data])
    except Exception as e:
        logger.warning(f"Issue in {binarizer_type} transformation: {e}")
        return np.zeros((1, len(binarizer.classes_)))

# Preprocess user data with robust handling
def preprocess_user_data(user_data):
    try:
        user_data_df = pd.DataFrame([user_data])

        # Handle missing or unexpected columns
        for col in ['Degree', 'Institute', 'age', 'Skills', 'Hobbies']:
            if col not in user_data_df.columns:
                logger.warning(f"Missing column '{col}', assigning default value.")
                user_data_df[col] = ['Unknown'] if col in ['Degree', 'Institute'] else [0 if col == 'age' else []]

        # Scale age
        age_scaled = scaler.transform(user_data_df[['age']])

        # Transform categorical data
        categorical_encoded = encoder.transform(user_data_df[['Degree', 'Institute']])

        # Safely transform skills and hobbies
        skills_encoded = safe_transform(skills_binarizer, user_data.get('Skills', []), 'skills')
        hobbies_encoded = safe_transform(hobbies_binarizer, user_data.get('Hobbies', []), 'hobbies')

        # Combine all features
        encoded_data = np.hstack([age_scaled, categorical_encoded, skills_encoded, hobbies_encoded])

        # Adjust feature size to match PCA expectations
        expected_features = pca.components_.shape[1]
        if encoded_data.shape[1] != expected_features:
            if encoded_data.shape[1] > expected_features:
                encoded_data = encoded_data[:, :expected_features]
            else:
                padding = np.zeros((encoded_data.shape[0], expected_features - encoded_data.shape[1]))
                encoded_data = np.hstack([encoded_data, padding])

        # Apply PCA
        reduced_data = pca.transform(encoded_data)
        return reduced_data
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return np.zeros((1, pca.n_components_))

# API endpoint for predicting cluster
@app.route('/predict-cluster', methods=['POST'])
def predict_cluster():
    try:
        user_data = request.json
        processed_data = preprocess_user_data(user_data)
        cluster = kmeans.predict(processed_data)
        return jsonify({'cluster': int(cluster[0])})
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'cluster': -1, 'message': 'Error occurred during prediction.'}), 500

# Run the Flask app in a separate thread
def run_app():
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 3000))
    app.run(host=host, port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    try:
        flask_thread = Thread(target=run_app)
        flask_thread.daemon = True
        flask_thread.start()
        logger.info("Flask app started successfully.")
        flask_thread.join()
    except KeyboardInterrupt:
        logger.info("Shutting down Flask app.")
