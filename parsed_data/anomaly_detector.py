import numpy as np
import joblib
import logging
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AnomalyDetector")


class AnomalyDetector:

    def __init__(
        self,
        model_path="best_log_model.keras",
        encoder_path="label_encoder.pkl"
    ):
        logger.info("Loading model...")
        self.model = load_model(model_path)

        logger.info("Loading encoder...")
        self.encoder = joblib.load(encoder_path)

        self.num_classes = len(
            self.encoder.classes_
        )

        logger.info(
            f"Loaded {self.num_classes} template classes."
        )

    def detect(
        self,
        sequence,
        actual_next_id,
        threshold=5.0,
        top_k=5
    ):
        """
        Detect anomalies using:
        1. Top-K DeepLog method
        2. Negative Log Likelihood score

        Parameters:
        ----------
        sequence : list
            Previous template IDs

        actual_next_id : int
            Actual next template ID

        threshold : float
            Score threshold

        top_k : int
            Number of top predictions

        Returns:
        --------
        dict
        """

        try:

            # Validate sequence
            if len(sequence) == 0:
                raise ValueError(
                    "Empty sequence provided."
                )

            # Unknown template detection
            if (
                actual_next_id < 0
                or
                actual_next_id >= self.num_classes
            ):
                logger.warning(
                    f"Unknown template ID "
                    f"{actual_next_id} detected."
                )

                return {
                    "anomaly": True,
                    "score": float("inf"),
                    "probability": 0.0,
                    "top_k_hit": False,
                    "top_predictions": [],
                    "note": "Unknown template"
                }

            # Prepare input
            input_seq = np.array(
                sequence,
                dtype=np.int32
            ).reshape(1, -1)

            # Predict
            probabilities = self.model.predict(
                input_seq,
                verbose=0
            )[0]

            # Top-K predictions
            top_predictions = np.argsort(
                probabilities
            )[-top_k:]

            top_k_hit = (
                actual_next_id
                in top_predictions
            )

            # Probability of actual event
            prob_actual = float(
                probabilities[
                    actual_next_id
                ]
            )

            # Negative Log Likelihood
            score = float(
                -np.log(
                    prob_actual + 1e-10
                )
            )

            # Final decision
            is_anomaly = (
                (not top_k_hit)
                or
                (score > threshold)
            )

            return {
                "anomaly": bool(
                    is_anomaly
                ),
                "score": score,
                "probability": prob_actual,
                "top_k_hit": bool(
                    top_k_hit
                ),
                "top_predictions":
                    top_predictions.tolist(),
                "predicted_template":
                    int(
                        np.argmax(
                            probabilities
                        )
                    )
            }

        except Exception as e:
            logger.error(
                f"Detection failed: {e}"
            )
            raise

def run_smoke_test():

    detector = AnomalyDetector(
        model_path="best_log_model.keras",
        encoder_path="label_encoder.pkl"
    )

    logger.info(
        "Smoke test successful. "
        "Model and encoder loaded."
    )


if __name__ == "__main__":
    run_smoke_test()  

    