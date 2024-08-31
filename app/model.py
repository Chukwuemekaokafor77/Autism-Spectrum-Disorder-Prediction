import joblib
import os

# Define the model path
model_path = os.path.join(os.getcwd(), 'model', 'model.pkl')

# Load the trained model
model = joblib.load(model_path)

# If you also want to load the scaler (optional)
scaler_path = os.path.join(os.getcwd(), 'model', 'scaler.pkl')
scaler = joblib.load(scaler_path)
