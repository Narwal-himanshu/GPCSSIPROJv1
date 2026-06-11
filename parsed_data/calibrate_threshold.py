import numpy as np
import pandas as pd
from feature_engineering import FeatureEngine
from anomaly_detector import AnomalyDetector

def calibrate():
    print("Loading data for calibration...")
    df = pd.read_csv("../parsed_data/parsed_logs.csv")
    engine = FeatureEngine(window_size=10)
    X, y = engine.transform(df, column_name="template_id")

    detector = AnomalyDetector("final_log_model.keras", "label_encoder.pkl")

    print("Calculating scores across the entire dataset...")
    all_scores = []
    
    # We iterate through all sequences to build our statistical baseline
    for i in range(1000):
        # We pass a very high threshold to ensure we capture the true scores
        res = detector.detect(X[i], int(y[i]), threshold=999.0)
        all_scores.append(res['score'])
        if i % 100 == 0:
           print(f"Processed {i}/1000")

    all_scores = np.array(all_scores)
    
    # Statistical baseline (Mean + 3 Standard Deviations)
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    new_threshold = mean_score + (3 * std_score)

    print(f"\n--- CALIBRATION RESULTS ---")
    print(f"Mean Anomaly Score: {mean_score:.4f}")
    print(f"Standard Deviation: {std_score:.4f}")
    print(f"Recommended Threshold (3σ): {new_threshold:.4f}")
    print(f"---------------------------")

if __name__ == "__main__":
    calibrate()