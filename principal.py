import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# Función para validar dataset cargado
# --------------------
def validar_dataset(df, target_col):
    # Revisar valores nulos
    if df.isnull().any().any():
        return False, "El dataset contiene valores nulos. Por favor límpialo antes de usarlo."

    # Revisar que target exista
    if target_col not in df.columns:
        return False, f"No se encontró la columna objetivo '{target_col}' en el dataset."

    # Revisar que todas las variables predictoras sean numéricas
    X = df.drop(columns=[target_col])
    if not np.all([np.issubdtype(dtype, np.number) for dtype in X.dtypes]):
        return False, "El dataset contiene variables no numéricas en las características."

    return True, "Dataset válido."

# --------------------
# Configuración inicial
# --------------------
st.set_page_config(page_title="Clasificación Interactiva", layout="wide")
st.title("🔍 Clasificación Interactiva de Datos")

# --------------------
# Sidebar
# --------------------
st.sidebar.header("⚙️ Configuración de Datos")

# Opción para cargar dataset
opcion_dataset = st.sidebar.radio("Fuente de datos", ["Generar dataset simulado", "Cargar CSV propio"])

if opcion_dataset == "Generar dataset simulado":
    n_samples = st.sidebar.slider("Número de muestras", 50, 1000, 200, 50)
    n_features = st.sidebar.slider("Número de características", 2, 20, 5, 1)
    n_informative = st.sidebar.slider("Características informativas", 1, n_features, 3, 1)
    n_classes = st.sidebar.slider("Número de clases", 2, 5, 2, 1)

    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_informative=n_informative, n_classes=n_classes,
                               random_state=42)
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(n_features)])
    df["Target"] = y
    target_col = "Target"

else:
    uploaded_file = st.sidebar.file_uploader("Sube tu CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.write("✅ Dataset cargado con éxito!")
        # Seleccionar columna objetivo
        target_col = st.sidebar.selectbox("Selecciona la columna objetivo (Target)", df.columns)
        valido, mensaje = validar_dataset(df, target_col)
        if not valido:
            st.error(mensaje)
            st.stop()
    else:
        st.warning("Por favor carga un archivo CSV para continuar.")
        st.stop()

# --------------------
# División del dataset
# --------------------
test_size = st.sidebar.slider("Proporción de test (%)", 10, 50, 20, 5) / 100
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# --------------------
# Preprocesamiento
# --------------------
scaling = st.sidebar.checkbox("Escalar datos", value=True)
if scaling:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# --------------------
# Selección de modelo
# --------------------
modelo = st.sidebar.selectbox("Modelo de Clasificación", ["Logistic Regression", "SVM", "Random Forest"])

if modelo == "Logistic Regression":
    C = st.sidebar.slider("C (Regularización)", 0.01, 10.0, 1.0)
    clf = LogisticRegression(C=C, max_iter=1000)
elif modelo == "SVM":
    C = st.sidebar.slider("C (Regularización)", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
    clf = SVC(C=C, kernel=kernel, probability=True)
else:
    n_estimators = st.sidebar.slider("Número de árboles", 10, 200, 100)
    max_depth = st.sidebar.slider("Profundidad máxima", 2, 20, 5)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

# --------------------
# Entrenamiento y predicción
# --------------------
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# --------------------
# Resultados
# --------------------
st.subheader("📊 Resultados del Modelo")

col1, col2 = st.columns(2)

with col1:
    st.write("### Reporte de Clasificación")
    st.text(classification_report(y_test, y_pred))

with col2:
    st.write("### Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

st.write("### Curva ROC")
fig, ax = plt.subplots()
if len(np.unique(y)) == 2:  # Binaria
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
else:  # Multiclase
    for i in range(y_proba.shape[1]):
        fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"Clase {i} (AUC = {roc_auc:.2f})")

ax.plot([0, 1], [0, 1], 'k--')
ax.legend()
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
st.pyplot(fig)

# --------------------
# Exploración de datos
# --------------------
st.subheader("🔎 Exploración de Datos")
if st.checkbox("Mostrar primeras filas"):
    st.write(df.head())

if st.checkbox("Mostrar estadísticas descriptivas"):
    st.write(df.describe())

if st.checkbox("Mostrar correlación entre variables"):
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --------------------
# Descargar dataset
# --------------------
st.download_button("📥 Descargar dataset en CSV", df.to_csv(index=False), "dataset.csv")
