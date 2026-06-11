import pandas as pd
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AnomalyInjector")


def inject_anomalies(
    df,
    column_name="template_id",
    num_to_inject=200,
    random_seed=42
):
    """
    Inject synthetic anomalies into a parsed log dataset.

    Parameters:
    ----------
    df : pd.DataFrame
        Parsed log dataframe.

    column_name : str
        Template ID column.

    num_to_inject : int
        Number of anomalies to inject.

    random_seed : int
        Reproducibility.

    Returns:
    -------
    pd.DataFrame
        Dataset with injected anomalies.
    """

    random.seed(random_seed)

    injected_df = df.copy()

    if column_name not in injected_df.columns:
        raise ValueError(
            f"Column '{column_name}' not found."
        )

    # Ground truth label
    injected_df["is_injected_anomaly"] = 0

    unique_templates = (
        injected_df[column_name]
        .unique()
        .tolist()
    )

    if len(unique_templates) < 2:
        raise ValueError(
            "Need at least 2 unique templates."
        )

    valid_indices = list(
        range(10, len(injected_df))
    )

    selected_indices = random.sample(
        valid_indices,
        min(num_to_inject, len(valid_indices))
    )

    for idx in selected_indices:

        original_template = (
            injected_df.loc[idx, column_name]
        )

        candidate_templates = [
            t for t in unique_templates
            if t != original_template
        ]

        new_template = random.choice(
            candidate_templates
        )

        injected_df.loc[
            idx,
            column_name
        ] = new_template

        injected_df.loc[
            idx,
            "is_injected_anomaly"
        ] = 1

    logger.info(
        f"Injected {len(selected_indices)} anomalies."
    )

    return injected_df


if __name__ == "__main__":

    INPUT_FILE = "../parsed_data/parsed_logs.csv"
    OUTPUT_FILE = "../parsed_data/dirty_logs.csv"

    df = pd.read_csv(INPUT_FILE)

    dirty_df = inject_anomalies(
        df,
        column_name="template_id",
        num_to_inject=200
    )

    dirty_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    logger.info(
        f"Saved dataset to {OUTPUT_FILE}"
    )

    print("\nDataset Summary")
    print("----------------")
    print(
        f"Total Rows: {len(dirty_df)}"
    )
    print(
        f"Injected Anomalies: "
        f"{dirty_df['is_injected_anomaly'].sum()}"
    )