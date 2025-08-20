import streamlit as st
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz

# --- Configuración de la página ---
st.set_page_config(page_title="ML Supervisado Demo", layout="wide")
st.title("🧠 Visualizador de Modelos Supervisados con Datos Simulados")

# --- Parámetros del dataset ---
st.sidebar.header("Configuración del Dataset")
n_samples = st.sidebar.slider("Número de muestras", 100, 5000, 1000)
n_features = st.sidebar.slider("Número de características", 2, 20, 5)
n_informative = st.sidebar.slider("Características informativas", 1, n_features, 3)
random_state = st.sidebar.number_input("Random seed", 0, 9999, 42)

# --- Cargar CSV personalizado ---
st.sidebar.header("Importar Dataset")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
use_custom_data = False

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset cargado exitosamente.")
        use_custom_data = True
    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {e}")

# --- Generar datos si no se subió un CSV ---
if not use_custom_data:
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

if "Target" not in df.columns:
    st.error("❗ El dataset debe contener una columna llamada 'Target' para entrenar los modelos supervisados.")
    st.stop()

X = df.drop("Target", axis=1)
y = df["Target"]

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
model_type = st.sidebar.selectbox("Selecciona un modelo", 
                                  ["Logistic Regression", "SVM", "Random Forest", "Decision Tree"])

if model_type == "Logistic Regression":
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

elif model_type == "SVM":
    model = SVC(probability=True)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

elif model_type == "Random Forest":
    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

elif model_type == "Decision Tree":
    st.sidebar.subheader("Parámetros del Árbol de Decisión")
    criterion = st.sidebar.selectbox("Criterio", ["gini", "entropy", "log_loss"])
    max_depth = st.sidebar.slider("Profundidad máxima", 1, 20, 5)
    min_samples_split = st.sidebar.slider("Mínimo de muestras para dividir un nodo", 2, 20, 2)

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    model.fit(X_train, y_train)  # Árbol no necesita escalado
    y_pred = model.predict(X_test)

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

# --- Importancia de características ---
if model_type in ["Random Forest", "Decision Tree"]:
    st.subheader("Importancia de las Características")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(feat_df.set_index("Feature"))

# --- Visualización del árbol de decisión ---
if model_type == "Decision Tree":
    st.subheader("🌳 Visualización del Árbol de Decisión")
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=X.columns,
        class_names=[str(c) for c in np.unique(y)],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.source.Source(dot_data)
    st.graphviz_chart(dot_data)

# --- Descargar dataset ---
st.download_button(
    label="⬇️ Descargar Dataset como CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="dataset.csv",
    mime="text/csv"
)
