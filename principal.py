import streamlit as st
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuración de la página ---
st.set_page_config(page_title="ML Supervisado Demo", layout="wide")
st.title("🧠 Visualizador de Modelos Supervisados con Datos Simulados")

# --- Parámetros del dataset ---
st.sidebar.header("Configuración del Dataset")
n_samples = st.sidebar.slider("Número de muestras", 100, 5000, 1000)
n_features = st.sidebar.slider("Número de características", 2, 20, 5)
n_informative = st.sidebar.slider("Características informativas", 1, n_features, 3)
random_state = st.sidebar.number_input("Random seed", 0, 9999, 42)

# --- Generar datos ---
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=0,
    n_classes=2,
    random_state=random_state,
)

df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
df["Target"] = y

st.subheader("Vista previa del Dataset")
st.dataframe(df.head())

# --- División en entrenamiento y prueba ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state
)

# --- Escalado de características ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Selección de modelo ---
st.sidebar.header("Modelo de ML")
model_type = st.sidebar.selectbox("Selecciona un modelo", ["Logistic Regression", "SVM", "Random Forest"])

if model_type == "Logistic Regression":
    model = LogisticRegression()
elif model_type == "SVM":
    model = SVC(probability=True)
else:
    model = RandomForestClassifier()

# --- Entrenamiento ---
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# --- Resultados ---
st.subheader("Métricas de Evaluación")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# --- Matriz de confusión ---
st.subheader("Matriz de Confusión")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
st.pyplot(fig)

# --- Importancia de características (si aplica) ---
if model_type == "Random Forest":
    st.subheader("Importancia de las Características")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": [f"Feature_{i}" for i in range(len(importances))],
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(feat_df.set_index("Feature"))
