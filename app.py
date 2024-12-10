#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from threading import Thread

# Initialize Flask app
app = Flask(__name__)

# Load the model and preprocessing objects
with open("preprocessing_objects.pkl", "rb") as f:
    objects = pickle.load(f)
encoder = objects['encoder']
skills_binarizer = objects['skills_binarizer']
hobbies_binarizer = objects['hobbies_binarizer']
scaler = objects['scaler']
pca = objects['pca']

with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Function to safely transform skills and hobbies
def safe_transform(binarizer, data, binarizer_type):
    try:
        # Adjust the binarizer's classes dynamically
        binarizer.classes_ = np.unique(np.concatenate([binarizer.classes_, data]))
        return binarizer.transform([data])
    except Exception as e:
        print(f"Warning: Issue in {binarizer_type} transformation - {e}")
        # Return a zero array if transformation fails
        return np.zeros((1, len(binarizer.classes_)))

# Preprocess user data with robust handling
def preprocess_user_data(user_data):
    try:
        # Convert input data to a DataFrame
        user_data_df = pd.DataFrame([user_data])

        # Handle missing or unexpected columns
        for col in ['Degree', 'Institute', 'age', 'Skills', 'Hobbies']:
            if col not in user_data_df.columns:
                print(f"Warning: Missing column '{col}', assigning default.")
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
        print(f"Error during preprocessing: {e}")
        # Return a default cluster input if preprocessing fails
        return np.zeros((1, pca.n_components_))

# API endpoint for predicting cluster
@app.route('/predict-cluster', methods=['POST'])
def predict_cluster():
    try:
        user_data = request.json

        # Preprocess the input data
        processed_data = preprocess_user_data(user_data)

        # Predict the cluster
        cluster = kmeans.predict(processed_data)
        return jsonify({'cluster': int(cluster[0])})
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Return a default cluster if something goes wrong
        return jsonify({'cluster': -1, 'message': 'Default cluster assigned due to an error'}), 200

# Run the Flask app in a separate thread
def run_app():
    app.run(port=3000, debug=False, use_reloader=False)

flask_thread = Thread(target=run_app)
flask_thread.start()


# In[97]:


import requests

# Sample user data
user_data = {
    "Degree": "Associate Degree",
    "Institute": "IIT Kanpur",
    "age": 20,
    "Skills": ['Embedded Systems'],
    "Hobbies": ["Writing"]
}

# Send a POST request to the Flask server
response = requests.post("http://127.0.0.1:3000/predict-cluster", json=user_data)

# Print the response from the API
print(response.json())


# In[ ]:




