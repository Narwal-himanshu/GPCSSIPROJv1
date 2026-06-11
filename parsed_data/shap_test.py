import shap
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from feature_engineering import FeatureEngine

print("Loading model...")
model = load_model("best_log_model.keras")

print("Loading data...")
df = pd.read_csv("../parsed_data/parsed_logs.csv")

engine = FeatureEngine(window_size=10)

X, y = engine.transform(
    df,
    column_name="template_id"
)

print(f"Dataset Shape: {X.shape}")

background = X[:20]

sample = X[100:101]

print("Creating prediction wrapper...")


def predict_fn(data):
    return model.predict(
        data,
        verbose=0
    )


print("Creating KernelExplainer...")

explainer = shap.KernelExplainer(
    predict_fn,
    background
)

print("Computing SHAP values...")

shap_values = explainer.shap_values(
    sample,
    nsamples=50
)

print("\nSUCCESS")

print(
    f"Type: {type(shap_values)}"
)

if isinstance(shap_values, list):

    print(
        f"Outputs: {len(shap_values)}"
    )

    print(
        f"Shape: "
        f"{np.array(shap_values[0]).shape}"
    )

else:

    print(
        f"Shape: "
        f"{np.array(shap_values).shape}"
    )