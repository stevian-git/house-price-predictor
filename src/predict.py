from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def get_neighborhoods(model):
    cats = model.named_steps["pre"].named_transformers_["cat"].categories_[0]
    return sorted(cats)


def predict_price(model, location, size, bedrooms, bathrooms, garage):
    row = pd.DataFrame([{
        "Neighborhood": location,
        "GrLivArea": size,
        "BedroomAbvGr": bedrooms,
        "Bathrooms": bathrooms,
        "GarageCars": garage,
    }])
    return float(np.expm1(model.predict(row)[0]))