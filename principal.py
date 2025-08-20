import streamlit as st
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuración de la página ---
st.set_page_config(page_title="ML Supervisado Demo", layout="wide")
st.title("🧠 Visualizador de Modelos Supervisados con Datos Simulados")

# --- Parámetros del dataset ---
st.sidebar.header("📊 Configuración del Dataset")
n_samples = st.sidebar.slider("Número de muestras", 100, 5000, 1000)
n_features = st.sidebar.slider("Número de características", 2, 20, 5)
n_informative = st.sidebar.slider("Características informativas", 1, n_features, 3)
random_state = st.sidebar.number_input("Random seed", 0, 9999, 42)
test_size = st.sidebar.slider("Proporción de Test (0.1 - 0.5)", 0.1, 0.5, 0.3)

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
eda_options = st.sidebar.multiselect("Opciones de EDA", [
    "Información básica",
    "Distribución Target",
    "Histogramas",
    "Correlación",
    "Dispersión",
    "Boxplot",
    "Pairplot"
])

if "Información básica" in eda_options:
    st.write("Shape:", df.shape)
    st.write(df.describe())

if "Distribución Target" in eda_options:
    fig, ax = plt.subplots()
    sns.countplot(x="Target", data=df, ax=ax)
    st.pyplot(fig)

if "Histogramas" in eda_options:
    feature = st.sidebar.selectbox("Selecciona feature (Histograma)", df.columns[:-1])
    bins = st.sidebar.slider("Bins (Histograma)", 5, 100, 20)
    fig, ax = plt.subplots()
    sns.histplot(df[feature], bins=bins, kde=True, ax=ax)
    st.pyplot(fig)

if "Correlación" in eda_options:
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

if "Dispersión" in eda_options:
    feat_x = st.sidebar.selectbox("Eje X (Scatter)", df.columns[:-1])
    feat_y = st.sidebar.selectbox("Eje Y (Scatter)", df.columns[:-1])
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[feat_x], y=df[feat_y], hue=df["Target"], palette="coolwarm", ax=ax)
    st.pyplot(fig)

if "Boxplot" in eda_options:
    feature = st.sidebar.selectbox("Feature (Boxplot)", df.columns[:-1], key="box")
    fig, ax = plt.subplots()
    sns.boxplot(x="Target", y=feature, data=df, palette="Set2", ax=ax)
    st.pyplot(fig)

if "Pairplot" in eda_options:
    sample_size = st.sidebar.slider("Muestras para Pairplot", 50, min(500, len(df)), 200)
    fig = sns.pairplot(df.sample(sample_size), hue="Target", diag_kind="kde")
    st.pyplot(fig)

# --- División en entrenamiento y prueba ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# --- Escalado de características (opcional) ---
scaling = st.sidebar.checkbox("Estandarizar características", True)
if scaling:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
else:
    X_train_scaled, X_test_scaled = X_train, X_test

# --- Selección de modelo ---
st.sidebar.header("🤖 Modelo de ML")
model_type = st.sidebar.selectbox("Selecciona un modelo", ["Logistic Regression", "SVM", "Random Forest"])

if model_type == "Logistic Regression":
    C = st.sidebar.slider("Regularización (C)", 0.01, 10.0, 1.0)
    model = LogisticRegression(C=C)
elif model_type == "SVM":
    kernel = st.sidebar.selectbox("Kernel SVM", ["linear", "rbf", "poly"])
    C = st.sidebar.slider("Regularización (C)", 0.01, 10.0, 1.0)
    model = SVC(probability=True, kernel=kernel, C=C)
else:
    n_estimators = st.sidebar.slider("Número de árboles", 10, 200, 100)
    max_depth = st.sidebar.slider("Profundidad máxima", 1, 20, 5)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

# --- Entrenamiento ---
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# --- Resultados ---
st.subheader("📈 Métricas de Evaluación")
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

# --- Curva ROC ---
st.subheader("Curva ROC")
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("Tasa Falsos Positivos")
ax.set_ylabel("Tasa Verdaderos Positivos")
ax.legend(loc="lower right")
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

# --- Extras ---
st.sidebar.header("⚙️ Extras")
if st.sidebar.checkbox("Exportar dataset a CSV"):
    st.download_button(
        label="Descargar Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dataset.csv",
        mime="text/csv"
    )

if st.sidebar.checkbox("Mostrar estadísticas avanzadas"):
    st.write(df.describe(include="all"))
