import pandas as pd
import logging
from ingestion import LogIngestor
from parsing import LogParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MainPipeline")


def run_pipeline():
    ingestor = LogIngestor("data")
    parser = LogParser()

    parsed_results = []

    logger.info("Starting log parsing pipeline...")

    for entry in ingestor.stream_logs():

        template_id = parser.process_log(entry["content"])

        parsed_results.append({
            "file": entry["file"],
            "line": entry["line"],
            "raw_log": entry["content"],
            "template_id": template_id
        })

    df = pd.DataFrame(parsed_results)

    # Generic name for any log source
    output_file = "parsed_data/parsed_logs.csv"

    df.to_csv(output_file, index=False)

    logger.info(
        f"Parsing complete. Saved {len(df)} records to {output_file}"
    )

    logger.info(
        f"Unique templates discovered: {df['template_id'].nunique()}"
    )


if __name__ == "__main__":
    run_pipeline()