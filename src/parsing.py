from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import logging

# Set up logging for better observability in production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LogParser")

class LogParser:
    def __init__(self):
        """
        Initializes the TemplateMiner with masking rules to ensure 
        proper clustering of dynamic log content.
        """
        config = TemplateMinerConfig()
        
        # Adjusting depth: 3-4 is usually optimal for general log patterns
        config.drain_depth = 3
        
        # Masking rules: These patterns are stripped before template extraction.
        # Customize these regex patterns to match the specific dynamic 
        # identifiers in your LANL dataset (User IDs, Computer IDs, etc.)
        config.masking = [
            {"regex_pattern": r"U\d+", "mask_with": "USER"},
            {"regex_pattern": r"C\d+", "mask_with": "COMPUTER"},
            {"regex_pattern": r"\d+", "mask_with": "NUM"}
        ]
        
        self.miner = TemplateMiner(config=config)
        logger.info("LogParser initialized with masking rules.")

    def process_log(self, log_content: str) -> int:
        """
        Processes a log line and returns the cluster ID.
        """
        try:
            result = self.miner.add_log_message(log_content)
            return result["cluster_id"]
        except Exception as e:
            logger.error(f"Error parsing log line: {e}")
            return -1 # Return -1 for failed parsing