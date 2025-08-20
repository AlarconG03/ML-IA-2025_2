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

# --- Exploratory Data Analysis (EDA) ---
st.header("🔍 Análisis Exploratorio de Datos (EDA)")

# Dimensiones y descripción
if st.checkbox("Mostrar información básica"):
    st.write("Shape:", df.shape)
    st.write(df.describe())

# Distribución de la variable objetivo
if st.checkbox("Distribución de la variable Target"):
    fig, ax = plt.subplots()
    sns.countplot(x="Target", data=df, ax=ax)
    st.pyplot(fig)

# Histogramas de features
if st.checkbox("Histogramas de características"):
    feature = st.selectbox("Selecciona una característica", df.columns[:-1])
    bins = st.slider("Número de bins", 5, 100, 20)
    fig, ax = plt.subplots()
    sns.histplot(df[feature], bins=bins, kde=True, ax=ax)
    st.pyplot(fig)

# Matriz de correlación
if st.checkbox("Matriz de correlación"):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Dispersión entre dos características
if st.checkbox("Gráfico de dispersión"):
    feat_x = st.selectbox("Eje X", df.columns[:-1])
    feat_y = st.selectbox("Eje Y", df.columns[:-1])
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[feat_x], y=df[feat_y], hue=df["Target"], palette="coolwarm", ax=ax)
    st.pyplot(fig)

# Boxplot
if st.checkbox("Boxplot de características"):
    feature = st.selectbox("Selecciona característica para boxplot", df.columns[:-1], key="box")
    fig, ax = plt.subplots()
    sns.boxplot(x="Target", y=feature, data=df, palette="Set2", ax=ax)
    st.pyplot(fig)

# Pairplot (muestras reducidas si el dataset es grande)
if st.checkbox("Pairplot entre variables"):
    sample_size = st.slider("Número de muestras a graficar (para no explotar el gráfico)", 50, min(500, len(df)), 200)
    fig = sns.pairplot(df.sample(sample_size), hue="Target", diag_kind="kde")
    st.pyplot(fig)

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
