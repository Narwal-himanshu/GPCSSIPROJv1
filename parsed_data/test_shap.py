import pandas as pd

from feature_engineering import FeatureEngine
from anomaly_detector import AnomalyDetector
from shap_explain import SHAPExplainer


def main():

    print("\n====================================")
    print("      SHAP EXPLAINABILITY TEST")
    print("====================================")

    print("\n[1] Loading parsed logs...")

    df = pd.read_csv(
        "../parsed_data/parsed_logs.csv"
    )

    print(
        f"Loaded {len(df)} log entries."
    )

    print("\n[2] Creating sequences...")

    engine = FeatureEngine(
        window_size=10
    )

    X, y = engine.transform(
        df,
        column_name="template_id"
    )

    print(
        f"Sequence Shape: {X.shape}"
    )

    print(
        f"Number of Classes: "
        f"{len(df['template_id'].unique())}"
    )

    print("\n[3] Loading trained model...")

    detector = AnomalyDetector(
        "best_log_model.keras",
        "label_encoder.pkl"
    )

    print(
        "Model loaded successfully."
    )

    print("\n[4] Initializing SHAP Explainer...")

    explainer = SHAPExplainer(
        detector.model,
        X[:1000],
        n_clusters=20
    )

    print(
        "SHAP initialized successfully."
    )

    print("\n[5] Selecting test sequence...")

    sequence = X[0]

    print(
        f"Sequence Length: "
        f"{len(sequence)}"
    )

    print(
        f"Sequence:\n{sequence}"
    )

    print("\n[6] Running anomaly detection...")

    result = detector.detect(
        sequence=sequence,
        actual_next_id=int(y[0]),
        threshold=15.67,
        top_k=20
    )

    print("\n===== DETECTION RESULT =====")

    print(
        f"Anomaly           : "
        f"{result['anomaly']}"
    )

    print(
        f"Score             : "
        f"{result['score']:.4f}"
    )

    print(
        f"Probability       : "
        f"{result['probability']:.6f}"
    )

    print(
        f"Top-K Hit         : "
        f"{result['top_k_hit']}"
    )

    print(
        f"Predicted Template: "
        f"{result['predicted_template']}"
    )

    print("\n[7] Running SHAP explanation...")

    explanation = explainer.explain(
        sequence,
        top_n=5
    )

    print(
        "\n===================================="
    )

    print(
        "      SHAP ROOT CAUSE ANALYSIS"
    )

    print(
        "===================================="
    )

    print(
        f"\nPredicted Template : "
        f"{explanation['predicted_template']}"
    )

    print(
        f"Confidence         : "
        f"{explanation['confidence']:.4f}"
    )

    print(
        "\nTop Contributors"
    )

    print(
        "---------------------------"
    )

    for item in explanation["top_contributors"]:

        print(
            f"\nRank #{item['rank']}"
        )

        print(
            f"Position    : "
            f"{item['position']}"
        )

        print(
            f"Template ID : "
            f"{item['template_id']}"
        )

        print(
            f"Importance  : "
            f"{item['importance']:.6f}"
        )

    print(
        "\n===== ROOT CAUSE SUMMARY ====="
    )

    print(
        explanation["root_cause"]
    )

    print(
        "\n===== TEST STATUS ====="
    )

    print(
        "SHAP explanation generated successfully."
    )

    print(
        "Explainable AI pipeline is operational."
    )


if __name__ == "__main__":
    main()