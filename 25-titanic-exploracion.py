import pandas as pd
import matplotlib.pyplot as plt

# Descarga el dataset directamente desde internet
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Guardalo en tu carpeta para no descargalo cada vez
df.to_csv("titatic.csv", index=False)

print("=== PRIMERAS 5 FILAS ===")
print(df.head())

print("\n=== INFORMACION DEL DATASET ===")
print(df.describe())

print("\n=== VALORES NULOS POR COLUMNA ===")
print(df.isnull().sum())
