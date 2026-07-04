import pandas as pd

df = pd.read_csv("titanic.csv")

print("=== ANTES DE LIMPIAR ===")
print(f"Filas y columnas: {df.shape}")
print(f"Valores nulos:\n{df.isnull().sum()}/n")

# 1. Edad: llenar los nulos con el promedio
promedio_edad = df["Age"].mean()
df["Age"] = df["Age"].fillna(promedio_edad)

#2. Embarked: llenar los 2 nulos con ek valor mas comun
valor_mas_comun = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(valor_mas_comun)

#3. Cabin: tiene 77% de nulos, mejor eliminar la columna
df = df.drop(columns=["Cabin"]) 

#4. Eliminar columnas que no sirben para predecir
df = df.drop(columns=["Name", "Ticket", "PassengerId"])

#5. Convertir texno a numeros (ML solo entiende numeros)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

print("=== DESPUES DE LIMPIAR ===")
print(f"Filas y columnas: {df.shape}")
print(f"Valores nulos:\n{df.isnull().sum()}/n")
print(df.head())

