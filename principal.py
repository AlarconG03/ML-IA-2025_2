import streamlit as st
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
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

# --- Vista previa ---
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
model_type = st.sidebar.selectbox(
    "Selecciona un modelo", 
    ["Logistic Regression", "SVM", "Random Forest"]
)

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

# --- EDA (Exploratory Data Analysis) ---
st.subheader("🔍 Análisis Exploratorio de Datos (EDA)")
st.markdown("Este análisis te permite explorar el dataset generado antes de aplicar modelos de Machine Learning.")

# --- Mostrar estadísticas descriptivas ---
if st.checkbox("📊 Mostrar estadísticas descriptivas"):
    st.write(df.describe())

# --- Selección de variables para análisis gráfico ---
st.sidebar.subheader("Opciones de EDA")
selected_feature = st.sidebar.selectbox("Selecciona una característica para análisis gráfico", df.columns[:-1])

# --- Histograma ---
if st.checkbox("📈 Mostrar histograma de una característica"):
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, ax=ax_hist, bins=20, color="skyblue")
    ax_hist.set_title(f"Distribución de {selected_feature}")
    st.pyplot(fig_hist)

# --- Boxplot ---
if st.checkbox("🧰 Mostrar boxplot por clase objetivo"):
    fig_box, ax_box = plt.subplots()
    sns.boxplot(data=df, x="Target", y=selected_feature, ax=ax_box, palette="Set2")
    ax_box.set_title(f"{selected_feature} por clase objetivo")
    st.pyplot(fig_box)

# --- Distribución de la variable objetivo ---
if st.checkbox("🧮 Mostrar distribución de clases"):
    st.write("Distribución de la variable objetivo:")
    class_counts = df["Target"].value_counts()
    st.bar_chart(class_counts)

# --- Mapa de calor de correlaciones ---
if st.checkbox("🔗 Mostrar mapa de calor de correlación"):
    corr_matrix = df.iloc[:, :-1].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Matriz de correlación entre características")
    st.pyplot(fig_corr)

# --- Scatterplot personalizado ---
if st.checkbox("📌 Mostrar scatterplot entre dos características"):
    col1 = st.sidebar.selectbox("X axis", df.columns[:-1], key="scatter_x")
    col2 = st.sidebar.selectbox("Y axis", df.columns[:-1], key="scatter_y")
    fig_scatter, ax_scatter = plt.subplots()
    sns.scatterplot(data=df, x=col1, y=col2, hue="Target", palette="cool", ax=ax_scatter)
    ax_scatter.set_title(f"{col1} vs {col2} por clase objetivo")
    st.pyplot(fig_scatter)

# --- Descargar dataset ---
st.download_button(
    label="⬇️ Descargar Dataset como CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="dataset.csv",
    mime="text/csv"
)

# ================================================
# 🌳 Árbol de Decisión
# ================================================
st.subheader("🌳 Árbol de Decisión")

target_col = st.selectbox("Selecciona la columna objetivo para el árbol", df.columns)

X_tree = df.drop(columns=[target_col])
y_tree = df[target_col]

# Detectar si es clasificación o regresión
if y_tree.dtype == "object" or len(np.unique(y_tree)) <= 10:
    is_classification = True
else:
    is_classification = False

# Parámetros ajustables en la barra lateral
st.sidebar.subheader("⚙️ Parámetros del Árbol de Decisión")
max_depth = st.sidebar.slider("Profundidad máxima", 1, 20, 3)
min_samples_split = st.sidebar.slider("Mínimo de muestras para dividir", 2, 20, 2)

if is_classification:
    criterion = st.sidebar.selectbox("Criterio de impureza", ["gini", "entropy", "log_loss"])
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=42
    )
else:
    criterion = st.sidebar.selectbox("Criterio de regresión", ["squared_error", "friedman_mse", "absolute_error", "poisson"])
    clf = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=42
    )

# Entrenar modelo
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y_tree, test_size=0.3, random_state=42)
clf.fit(X_train_tree, y_train_tree)

# Predicciones
y_pred_tree = clf.predict(X_test_tree)

# Mostrar métricas
if is_classification:
    st.write("**Accuracy:**", accuracy_score(y_test_tree, y_pred_tree))
    st.text("Reporte de clasificación:")
    st.text(classification_report(y_test_tree, y_pred_tree))
else:
    st.write("**MSE (Error cuadrático medio):**", np.mean((y_test_tree - y_pred_tree) ** 2))

# Graficar el árbol
st.write("### Visualización del Árbol")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(clf, feature_names=X_tree.columns, filled=True, fontsize=8, ax=ax)
st.pyplot(fig)
