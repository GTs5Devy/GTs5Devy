import pandas as pd
df = pd.read_csv("data/dry_laps.csv", delimiter=";")
print(df.columns.tolist())
print(df.head(2))