# Análisis Predictivo de Supervivencia - Titanic 🚢

Proyecto de análisis de datos con Python sobre el clásico dataset del Titanic (Kaggle), aplicando el flujo completo de un proyecto de ciencia de datos: exploración, limpieza, visualización y modelado predictivo.

## 🎯 Objetivo

Analizar los datos de los pasajeros del Titanic para identificar qué factores influyeron en su supervivencia, y entrenar un modelo de Machine Learning capaz de predecirla.

## 🛠️ Herramientas utilizadas

- **Python**
- **Pandas** — limpieza y manipulación de datos
- **Matplotlib** — visualización de datos
- **Scikit-learn** — modelo de Machine Learning (Random Forest)

## 📂 Estructura del proyecto

| Archivo | Descripción |
|---|---|
| `25-titanic-exploracion.py` | Carga del dataset y diagnóstico inicial (valores nulos, estadísticas descriptivas) |
| `26-titanic-limpieza.py` | Limpieza de datos: valores nulos, columnas irrelevantes, conversión de texto a números |
| `27-titanic-modelo.py` | Entrenamiento y evaluación de un modelo Random Forest |
| `28-titanic-graficas.py` | Visualización de patrones de supervivencia |

## 🔍 Proceso

1. **Exploración:** el dataset contiene 891 pasajeros y 12 columnas, con valores nulos en `Age`, `Cabin` y `Embarked`.
2. **Limpieza:**
   - `Age`: nulos completados con el promedio.
   - `Embarked`: nulos completados con el valor más frecuente.
   - `Cabin`: eliminada (77% de datos faltantes).
   - `Name`, `Ticket`, `PassengerId`: eliminadas por no aportar valor predictivo.
   - `Sex` y `Embarked`: convertidas de texto a valores numéricos.
3. **Modelado:** entrenamiento de un Random Forest (80% entrenamiento / 20% prueba).
4. **Visualización:** 4 gráficas mostrando supervivencia por género, clase, edad e importancia de variables.

## 📊 Resultados

- **Precisión del modelo:** 82.12%
- **Variables más determinantes:** tarifa pagada (`Fare`), sexo (`Sex`) y edad (`Age`)
- Los resultados coinciden con el contexto histórico del hundimiento: se priorizó la evacuación de mujeres, niños y pasajeros de clases más altas.

![Gráficas del análisis](graficas_titanic.png)

## 💡 Aplicación práctica

El mismo proceso de este proyecto —limpieza, análisis exploratorio, visualización y modelado— es aplicable a datasets reales de negocio: análisis de clientes, predicción de abandono (*churn*), análisis de ventas, o generación de reportes automáticos.

---

📌 Proyecto realizado como parte de mi proceso de aprendizaje en análisis de datos con Python.
