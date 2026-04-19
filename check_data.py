import pandas as pd

df = pd.read_csv("data/sales.csv", encoding="latin1")

print(df.head())
print("\nKolumny:")
print(df.columns.tolist())
print("\nShape:")
print(df.shape)