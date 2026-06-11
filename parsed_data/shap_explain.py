import shap
import numpy as np
import logging

from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SHAPExplainer")


class SHAPExplainer:

    def __init__(
        self,
        model,
        training_data,
        n_clusters=20
    ):
        """
        SHAP Kernel Explainer using K-Means centroids
        as background data.

        Parameters
        ----------
        model : keras model
            Trained anomaly detection model

        training_data : np.ndarray
            Full training sequence dataset (X)

        n_clusters : int
            Number of background centroids
        """

        logger.info(
            "Initializing SHAP Explainer..."
        )

        self.model = model

        logger.info(
            f"Generating {n_clusters} "
            f"background centroids..."
        )

        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=42
        )

        kmeans.fit(training_data)

        background = (
            kmeans.cluster_centers_
        )

        logger.info(
            f"Background shape: "
            f"{background.shape}"
        )

        self.explainer = (
            shap.KernelExplainer(
                self._predict,
                background
            )
        )

        logger.info(
            "SHAP Explainer ready."
        )

    def _predict(
        self,
        data
    ):
        """
        Prediction wrapper required by SHAP.
        """

        try:

            predictions = (
                self.model.predict(
                    data,
                    verbose=0
                )
            )

            return predictions

        except Exception as e:

            logger.error(
                f"Prediction error: {e}"
            )

            return np.zeros(
                (
                    data.shape[0],
                    self.model.output_shape[-1]
                )
            )

    def explain(
        self,
        sequence,
        top_n=5
    ):
        """
        Explain a single sequence.

        Parameters
        ----------
        sequence : np.ndarray
            Shape: (10,)

        top_n : int
            Number of top contributors

        Returns
        -------
        list
        """

        sequence = np.array(
            sequence
        ).reshape(
            1,
            -1
        )

        logger.info(
            "Calculating SHAP values..."
        )

        shap_values = (
            self.explainer.shap_values(
                sequence,
                nsamples=50
            )
        )

        # Shape observed:
        # (1, 10, 475)
        #
        # 1 = sample
        # 10 = sequence positions
        # 475 = template classes

        importance = np.mean(
            np.abs(
                shap_values[0]
            ),
            axis=1
        )

        top_indices = (
            np.argsort(
                importance
            )[::-1][:top_n]
        )

        explanation = []

        for idx in top_indices:

            explanation.append(
                {
                    "position":
                        int(idx),

                    "template_id":
                        int(
                            sequence[0][idx]
                        ),

                    "importance":
                        float(
                            importance[idx]
                        )
                }
            )

        return explanation

    def print_explanation(
        self,
        explanation
    ):
        """
        Pretty print SHAP explanation.
        """

        print(
            "\n===== SHAP EXPLANATION ====="
        )

        for item in explanation:

            print(
                f"Position: "
                f"{item['position']} | "
                f"Template: "
                f"{item['template_id']} | "
                f"Importance: "
                f"{item['importance']:.6f}"
            )