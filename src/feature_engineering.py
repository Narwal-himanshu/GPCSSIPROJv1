import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Tuple

# Configure logging for observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FeatureEngineering")


class FeatureEngine:
    """
    Handles encoding and sliding window sequence generation
    for log template IDs.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.encoder = LabelEncoder()

        logger.info(
            f"FeatureEngine initialized with window_size={window_size}"
        )

    def transform(
        self,
        df: pd.DataFrame,
        column_name: str = "template_id"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encodes template IDs and transforms them into
        sliding window sequences for LSTM training.

        Args:
            df: Structured DataFrame from parsing layer.
            column_name: Target column containing template IDs.

        Returns:
            X: Input sequences
            y: Target values
        """

        if column_name not in df.columns:
            logger.error(
                f"Column '{column_name}' not found in dataframe."
            )
            raise ValueError(
                f"Feature engineering failed: '{column_name}' missing."
            )

        try:
            # Remove missing values
            df = df.dropna(subset=[column_name])

            # Encode categorical template IDs
            encoded_data = self.encoder.fit_transform(
                df[column_name]
            )
             
            logger.info(f"Encoded shape: {encoded_data.shape}")
            logger.info(f"First encoded value: {encoded_data[0]}")
            logger.info(f"Type: {type(encoded_data[0])}")

            # Ensure enough data exists
            if len(encoded_data) <= self.window_size:
                raise ValueError(
                    f"Data length ({len(encoded_data)}) "
                    f"is too short for window_size={self.window_size}"
                )

            sequences = []
            targets = []

            # Sliding Window Generation
            for i in range(
                len(encoded_data) - self.window_size
            ):
                sequences.append(
                    encoded_data[i:i + self.window_size]
                )

                targets.append(
                    encoded_data[i + self.window_size]
                )

            X = np.array(sequences, dtype=np.int32)
            y = np.array(targets, dtype=np.int32)

            # LSTM expects:
            # (samples, timesteps, features)
           # X = X.reshape(
          #      X.shape[0],
         #   1
         #   )

            logger.info(
                f"Feature engineering successful."
            )
            logger.info(
                f"Unique templates: {df[column_name].nunique()}"
            )
            logger.info(
                f"Sequence shape: {X.shape}"
            )

            return X, y

        except Exception as e:
            logger.error(
                f"Error during sequence transformation: {e}"
            )
            raise