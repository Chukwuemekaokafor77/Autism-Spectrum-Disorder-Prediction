import joblib
import os

# Define paths relative to the project root (one level above this file's directory)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

model_path = os.path.join(MODEL_DIR, 'model.pkl')
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')


def _ensure_artifacts():
    """Ensure model artifacts exist; train them if missing.

    On platforms like Render free tier where we cannot run a pre-deploy
    command, this allows the app to bootstrap itself by training the
    model at startup if necessary.
    """

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return

    # Train the model and create the artifacts
    try:
        from train import train_and_save

        train_and_save()
    except Exception:
        # If training fails, we let the subsequent load raise a clear error
        pass


def _load_artifact(path):
    """Internal helper to load a joblib artifact from disk."""
    return joblib.load(path)


# Ensure artifacts exist, then load the trained model and scaler at import time
_ensure_artifacts()
model = _load_artifact(model_path)
scaler = _load_artifact(scaler_path)
