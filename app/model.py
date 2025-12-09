import joblib
import os

# Define paths relative to the project root (one level above this file's directory)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

model_path = os.path.join(MODEL_DIR, 'model.pkl')
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')


def _load_artifact(path):
    """Internal helper to load a joblib artifact from disk."""
    return joblib.load(path)


# Load the trained model and scaler at import time for fast inference
model = _load_artifact(model_path)
scaler = _load_artifact(scaler_path)
