import pandas as pd

df = pd.read_csv("data/spam.csv", sep="\t", header=None, names=["label", "text"])

print("Aperçu du dataset :")
print(df.head())

print("\nNombre de lignes :", len(df))

print("\nRépartition des classes :")
print(df["label"].value_counts())