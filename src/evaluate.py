import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from anomaly_detector import AnomalyDetector
from feature_engineering import FeatureEngine


def evaluate():

    print("Loading dirty dataset...")

    df = pd.read_csv(
        "parsed_data/dirty_logs.csv"
    )

    # Ground truth labels
    y_true = (
        df["is_injected_anomaly"]
        .values[10:8010]
    )

    engine = FeatureEngine(
        window_size=10
    )

    X, y = engine.transform(
        df,
        column_name="template_id"
    )

    print(
        f"Total sequences: {len(X)}"
    )

    print(
        f"Injected anomalies: "
        f"{sum(y_true)}"
    )

    detector = AnomalyDetector(
        "models/best_log_model.keras",
        "models/label_encoder.pkl"
    )

    y_pred = []

    print(
        "\nEvaluating detector..."
    )

    for i in range(min(8000, len(X))):

        if i % 1000 == 0:
            print(
                f"Processed "
                f"{i}/{len(X)}"
            )

        result = detector.detect(
            sequence=X[i],
            actual_next_id=int(y[i]),
            threshold=15.67,
            top_k=20
        )

        y_pred.append(
            int(
                result["anomaly"]
            )
        )

    print(
        "\n===== CONFUSION MATRIX ====="
    )

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

    print(
        "\n===== CLASSIFICATION REPORT ====="
    )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Normal",
                "Anomaly"
            ]
        )
    )


if __name__ == "__main__":
    evaluate()