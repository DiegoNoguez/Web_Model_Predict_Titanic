import joblib
import json
import os
from django.conf import settings

def load_resources():
    base_dir = settings.BASE_DIR
    
    model_path = os.path.join(base_dir, 'model.pkl')
    scaler_path = os.path.join(base_dir, 'scaler.pkl')
    features_path = os.path.join(base_dir, 'features.json')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(features_path, "r") as f:
        features = json.load(f)
    
    return model, scaler, features

# Carga global al iniciar la app
model, scaler, features = load_resources()