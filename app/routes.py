import pandas as pd
import joblib
import os

# app/routes.py
from flask import Blueprint, render_template, request
from .model import model, scaler  # Ensure scaler is loaded if needed

# Load the feature names used during training
feature_names_path = os.path.join(os.getcwd(), 'model', 'feature_names.pkl')
model_columns = joblib.load(feature_names_path)

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Map the form inputs to variables corresponding to the updated form fields
            data = {
                'A1': float(request.form.get('A1', 0)),
                'A2': float(request.form.get('A2', 0)),
                'A3': float(request.form.get('A3', 0)),
                'A4': float(request.form.get('A4', 0)),
                'A5': float(request.form.get('A5', 0)),
                'A6': float(request.form.get('A6', 0)),
                'A7': float(request.form.get('A7', 0)),
                'A8': float(request.form.get('A8', 0)),
                'A9': float(request.form.get('A9', 0)),
                'A10': float(request.form.get('A10', 0)),
                'Age_Mons': float(request.form.get('Age_Mons', 0)),
                'Sex': request.form.get('Sex', '0'),
                'Ethnicity': request.form.get('Ethnicity', 'Others'),
                'Jaundice': request.form.get('Jaundice', '0'),
                'Family_mem_with_ASD': request.form.get('Family_mem_with_ASD', '0')
            }

            # Convert the input data to a DataFrame
            data_df = pd.DataFrame([data])

            # One-hot encode the categorical features to match the training phase
            data_df = pd.get_dummies(data_df)

            # Align the DataFrame to ensure all feature columns match those during training
            data_df = data_df.reindex(columns=model_columns, fill_value=0)

            # Scale the input data
            data_scaled = scaler.transform(data_df)

            # Make prediction
            prediction = model.predict(data_scaled)[0]
            result = 'Positive for ASD Traits' if prediction == 1 else 'Negative for ASD Traits'
            
            return render_template('index.html', result=result)
        except Exception as e:
            return render_template('index.html', result=f"An error occurred: {e}")
    
    return render_template('index.html')
