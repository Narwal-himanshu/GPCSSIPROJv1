import pandas as pd
import logging
import joblib
from feature_engineering import FeatureEngine
from model_trainer import build_and_train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainingPipeline")


def train():
    logger.info("Loading parsed logs from CSV...")

    df = pd.read_csv("parsed_data/parsed_logs.csv")

    engine = FeatureEngine(window_size=10)

    X, y = engine.transform(
        df,
        column_name="template_id"
    )

    logger.info(
        f"Data ready for training. X shape: {X.shape}, y shape: {y.shape}"
    )

    num_classes = len(engine.encoder.classes_)

    logger.info(
        f"Number of classes detected by encoder: {num_classes}"
    )

    logger.info(f"Min label: {y.min()}")
    logger.info(f"Max label: {y.max()}")

    logger.info("Starting model training...")

    model, history = build_and_train_model(
        X,
        y,
        num_classes=num_classes
    )

    joblib.dump(
        engine.encoder,
        "models/label_encoder.pkl"
    )

    logger.info("Model training completed.")

    model.save("models/final_log_model.keras")

    return model, history


if __name__ == "__main__":
    train()