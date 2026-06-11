
import pandas as pd

from feature_engineering import FeatureEngine

from anomaly_detector import AnomalyDetector


df = pd.read_csv("../parsed_data/parsed_logs.csv")

engine = FeatureEngine(window_size=10)

X, y = engine.transform(
    df,
    column_name="template_id"
)

detector = AnomalyDetector(
    "best_log_model.keras",
    "label_encoder.pkl"
)

result = detector.detect(
    sequence=X[0],
    actual_next_id=int(y[0]),
    threshold=8.0,
    top_k=20
)

print("\n===== DETECTION RESULT =====")
print(f"Anomaly: {result['anomaly']}")
print(f"Score: {result['score']:.4f}")
print(f"Probability: {result['probability']:.6f}")
print(f"Top-K Hit: {result['top_k_hit']}")
print(f"Predicted Template: {result['predicted_template']}")

print("\n===== BATCH TEST =====")

anomalies = 0
top_k_misses = 0
for i in range(100):

    result = detector.detect(
        sequence=X[i],
        actual_next_id=int(y[i]),
        threshold=8.0,
        top_k=20
    )
    if not result["top_k_hit"]:
        top_k_misses += 1


    if result["anomaly"]:
        anomalies += 1

print(f"Sequences Tested: 100")
print(f"Anomalies Detected: {anomalies}")
print(f"Normal Sequences: {100 - anomalies}")
print(f"Top-K Misses: {top_k_misses}")