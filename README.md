# Log Anomaly Detector

This repository contains an AI-powered log anomaly detection pipeline designed to read through system logs to understand and detect abnormal logging behaviors or inconsistent data leaks.

The system uses a custom deep learning model (LSTM) and an explainable AI approach (SHAP) combined with automated log parsing (Drain3) to identify when anomalous sequences of log events occur in the system.

## Project Structure

- **`src/`**: Contains the core Python scripts for ingestion, parsing, feature engineering, model training, evaluation, and anomaly detection.
- **`models/`**: Stores the trained machine learning model files (`.keras`) and label encoders (`.pkl`).
- **`data/`**: Directory containing raw system log files used for ingestion.
- **`parsed_data/`**: Directory where parsed outputs and synthetic datasets are stored.
- **`frontend/`**: Contains a simple web interface (`index.html`, `style.css`, `script.js`) for users to upload logs and visualize anomalies.

## Custom AI vs. LLMs for Log Analysis

When deciding between building a custom AI model (like the one in this repo) or using a Large Language Model (LLM) for log anomaly detection, consider the structure of your data:

### Structured vs. Unstructured Logs
- **Custom AI / Machine Learning (Current Approach):**
  Building a custom AI pipeline (e.g., parsing logs into templates with Drain, and using an LSTM for sequence prediction) is highly effective for logs that can be structured or semi-structured. It learns the exact sequential patterns of *your* specific system. This approach is significantly faster at inference time, cheaper to run, and less prone to hallucination compared to LLMs.
- **Large Language Models (LLMs):**
  LLMs shine when dealing with highly unstructured, diverse, and human-readable text logs where traditional template mining fails. They are excellent at understanding the *context* or *semantic meaning* of a totally new, unseen error message.

### Recommendation
For a production system where performance, cost, and strict predictability are key, a **custom AI in Python** (the current approach) is generally recommended, provided you use an effective parser (like Drain3) to convert semi-structured logs into structured sequence data.

If your logs are completely unstructured, highly variable, and require semantic understanding (e.g., analyzing natural language user feedback mixed with error codes), integrating an **LLM** might be more beneficial, though it will come with higher computational costs.
