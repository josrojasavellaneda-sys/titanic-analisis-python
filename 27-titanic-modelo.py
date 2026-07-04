import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Cargar y limpiar datos(igual que antes)
df = pd.read_csv("titanic.csv")
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Separar X (lo que el modelo usa para predecir) y y (lo que qureremos predecir)
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Train/Test Split: 80% entrenar, 20% probar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Datos de entrenamiento: {len(X_train)} pasajeros")
print(f"Datos de prueba: {len(X_test)} pasajeros")

# Entrenar el modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Predecir en los datos de pruena
predicciones = modelo.predict(X_test)

# Medir precision
precision = accuracy_score(y_test, predicciones)
print(f"\nPrecision del modelo: {precision:.2%}")
print("\n=== REPORTE DETALLADO ===")
print(classification_report(y_test, predicciones, target_names=["Murio", "Sobrevivio"]))

# Que variables importaron mas para predecir 
importancias = pd.Series(modelo.feature_importances_, index=X.columns)
importancias = importancias.sort_values(ascending=False)
print("=== VARIABLES MAS IMPORTANTES ===")
print(importancias)
