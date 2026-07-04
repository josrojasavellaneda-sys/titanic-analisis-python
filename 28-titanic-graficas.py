import pandas as pd
import matplotlib.pyplot as plt

# Cargar y limpiar datos
df = pd.read_csv("titanic.csv")
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Analisis de supervivencia - Titanic", fontsize=16, fontweight="bold")

# Grafica 1: Supervivencia por genero 
sobrevivieron = df[df["Survived"] == 1]
murieron = df[df["Survived"] == 0]
generos =["Hombre", "Mujer"]
sv_genero = [sobrevivieron[sobrevivieron["Sex"] ==0]["Sex"].count(),
             sobrevivieron[sobrevivieron["Sex"] == 1]["Sex"].count()]
mu_genero = [murieron[murieron["Sex"] == 0]["Sex"].count(),
             murieron[murieron["Sex"] == 1]["Sex"].count()]

x = range(len(generos))
ax1.bar(x, mu_genero, label="Murio", color="salmon")
ax1.bar(x, sv_genero, bottom=mu_genero, label="Sobrevivio,", color="steelblue")
ax1.set_xticks(x)
ax1.set_xticklabels(generos)
ax1.set_title("Supervivencia por genero")
ax1.set_ylabel("Pasajeros")
ax1.legend()

# Grafica 2: Supervivencia por clase
clases = [1, 2, 3]
sv_clase = [sobrevivieron[sobrevivieron["Pclass"] == c]["Pclass"].count() for c in clases]
mu_clase = [murieron[murieron["Pclass"] == c]["Pclass"].count() for c in clases]

x2 = range(len(clases))
ax2.bar(x2, mu_clase, label="Murio", color="salmon")
ax2.bar(x2, sv_clase, bottom=mu_clase, label="Sobrevivió", color="steelblue")
ax2.set_xticks(x2)
ax2.set_xticklabels(["Primera", "Segunda", "Tercera"])
ax2.set_title("Spuervivencia por clase")
ax2.set_ylabel("pasajeros")
ax2.legend()
ax2.set_ylim(0, 550)


# Grafica 3: Distribucion de edades (sobrevivieron vs murieron)
ax3.hist(murieron["Age"], bins=20, alpha=0.6, color="salmon", label="Murio")
ax3.hist(sobrevivieron["Age"], bins=20, alpha=0.6, color="steelblue", label="Sobrevivieron")
ax3.set_title("Distribucion de edades")
ax3.set_xlabel("Edad")
ax3.set_ylabel("Cantidad")
ax3.legend()
df_edad = pd.read_csv("titanic.csv")


# Grafica 4: importancia de variables
importancias = {"fare": 0.272, "Sex": 0.266, "Age": 0.252,
                "Pclass": 0.085, "SibSp": 0.053, "Parch": 0.037, "Embarked": 0.034}
ax4.barh(list(importancias.keys()), list(importancias.values()), color="steelblue")
ax4.set_title("Variables mas importantes para el modelo")
ax4.set_xlabel("importancia")

plt.tight_layout()
plt.show()
