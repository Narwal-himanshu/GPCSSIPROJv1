import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IngestionEngine")

class LogIngestor:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path

    def stream_logs(self):
        """Streams logs line by line to maintain low memory footprint."""
        for filename in os.listdir(self.raw_data_path):
            file_path = os.path.join(self.raw_data_path, filename)
            try:
                with open(
                           file_path,
                           "r",
                          encoding="utf-8",
                          errors="replace"
                                           ) as f:
                    for line_no, line in enumerate(f, start=1):
                        yield {"file": filename, "line": line_no, "content": line.strip()}
            except Exception as e:
                logger.error(f"Failed to read {filename}: {e}")