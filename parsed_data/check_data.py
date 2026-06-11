import pandas as pd

df = pd.read_csv("../parsed_data/parsed_logs.csv")

print("\nTop 20 Templates:\n")
print(df["template_id"].value_counts().head(20))