# ============================================================
# CASO DE ESTUDIO: RECURSOS HUMANOS
# Predicción de rotación de empleados
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt          
from sklearn.base import clone           # Copia modelos sin entrenar

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Evaluación
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. DATASET
# ============================================================

# renuncia:
# 0 = permanece
# 1 = renuncia

data = {
    "edad": [
        25, 28, 35, 40, 30, 45, 32, 29, 50, 38,
        27, 33, 41, 36, 31, 48, 26, 39, 42, 34
    ],
    "salario": [
        1800, 2000, 3500, 4500, 2200, 5000, 2800, 2100, 6000, 4000,
        1900, 3000, 4700, 3600, 2500, 5500, 1700, 4200, 4800, 3200
    ],
    "años_empresa": [
        1, 2, 5, 8, 2, 10, 4, 2, 15, 7,
        1, 4, 9, 6, 3, 12, 1, 8, 10, 5
    ],
    "satisfaccion": [
        2, 3, 7, 8, 3, 9, 5, 2, 9, 7,
        2, 6, 8, 6, 4, 9, 1, 7, 8, 5
    ],
    "horas_extra": [
        15, 12, 5, 4, 14, 3, 8, 13, 2, 6,
        16, 7, 4, 6, 10, 3, 18, 5, 4, 8
    ],
    "renuncia": [
        1, 1, 0, 0, 1, 0, 0, 1, 0, 0,
        1, 0, 0, 0, 1, 0, 1, 0, 0, 0
    ]
}

df = pd.DataFrame(data)

print("Dataset inicial:")
print(df.head())

# Matriz de correlación
corr = df.corr()

fig_corr, ax_corr = plt.subplots(figsize=(7, 6))
imagen = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

ax_corr.set_xticks(range(len(corr.columns)))
ax_corr.set_yticks(range(len(corr.columns)))
ax_corr.set_xticklabels(corr.columns, rotation=45, ha="right")
ax_corr.set_yticklabels(corr.columns)

for i in range(len(corr)):
    for j in range(len(corr)):
        ax_corr.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

fig_corr.colorbar(imagen, label="Coeficiente de correlación")
ax_corr.set_title("Matriz de correlación")
fig_corr.tight_layout()
plt.show()

# ============================================================
# 2. VARIABLES X e y
# ============================================================

X = df[
    [
        "edad",
        "salario",
        "años_empresa",
        "satisfaccion",
        "horas_extra"
    ]
]

y = df["renuncia"]

# ============================================================
# 3. DIVISIÓN DE DATOS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ============================================================
# 4. MODELOS
# ============================================================

modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(kernel="linear"),
    "Naive Bayes": GaussianNB()
}

# ============================================================
# 5. EVALUACIÓN DE MODELOS
# ============================================================

resultados = []

# Matriz de confusión por modelo
fig_cm, ejes_cm = plt.subplots(2, 3, figsize=(12, 7))
ejes_cm = ejes_cm.ravel()

for i, (nombre, modelo) in enumerate(modelos.items()):

    print("\n" + "="*70)
    print("MODELO:", nombre)
    print("="*70)

    # Entrenamiento
    modelo.fit(X_train, y_train)

    # Predicción
    y_pred = modelo.predict(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Validación cruzada
    cv_scores = cross_val_score(
        modelo,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    cv_promedio = cv_scores.mean()

    resultados.append({
        "Modelo": nombre,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Validación cruzada": cv_promedio
    })

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Validación cruzada:", cv_scores)
    print("Promedio:", cv_promedio)

    # Gráfica de la matriz de confusión del modelo
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    ejes_cm[i].imshow(cm, cmap="Greens")
    for fila, col, etiqueta in [(0, 0, "TN"), (0, 1, "FP"), (1, 0, "FN"), (1, 1, "TP")]:
        ejes_cm[i].text(col, fila, f"{etiqueta}\n{cm[fila, col]}",
                        ha="center", va="center", fontsize=12)
    ejes_cm[i].set_xticks([0, 1])
    ejes_cm[i].set_yticks([0, 1])
    ejes_cm[i].set_xticklabels(["Permanece", "Renuncia"])
    ejes_cm[i].set_yticklabels(["Permanece", "Renuncia"])
    ejes_cm[i].set_xlabel("Predicho")
    ejes_cm[i].set_ylabel("Real")
    ejes_cm[i].set_title(nombre)

# Mostrar las matrices de confusión de todos los modelos
fig_cm.suptitle("Matrices de confusión")
fig_cm.tight_layout()
plt.show()

# ============================================================
# 6. TABLA FINAL
# ============================================================

df_resultados = pd.DataFrame(resultados)

print("\n" + "="*70)
print("TABLA COMPARATIVA")
print("="*70)

print(df_resultados.sort_values(by="F1-score", ascending=False))

# Gráfica comparativa de métricas por modelo
df_resultados.set_index("Modelo").plot(kind="bar", figsize=(12, 5), ylim=(0, 1.1))
plt.title("Comparación de métricas por modelo")
plt.ylabel("Valor de la métrica")
plt.xticks(rotation=15)
plt.legend(loc="lower left", ncol=5)
plt.tight_layout()
plt.show()

# ============================================================
# 7. MEJOR MODELO
# ============================================================

mejor_modelo = df_resultados.sort_values(
    by="F1-score",
    ascending=False
).iloc[0]

print("\nMejor modelo según F1-score:")
print(mejor_modelo)

# ============================================================
# 8. ESCENARIO AMPLIADO
# Se agregan: modalidad_trabajo y liderazgo
# ============================================================

df_ext = df.copy()

# liderazgo: percepción del liderazgo del jefe (1 = malo, 10 = excelente)
df_ext["liderazgo"] = [
    3, 4, 7, 8, 3, 9, 5, 3, 9, 7,
    2, 6, 8, 6, 4, 9, 2, 7, 8, 5
]

# modalidad_trabajo: presencial / hibrido / remoto
df_ext["modalidad_trabajo"] = [
    "presencial", "presencial", "hibrido", "remoto", "presencial",
    "remoto", "hibrido", "presencial", "remoto", "hibrido",
    "presencial", "hibrido", "remoto", "hibrido", "presencial",
    "remoto", "presencial", "hibrido", "remoto", "hibrido"
]

# Variable categórica -> numérica (One-Hot)
df_ext = pd.get_dummies(df_ext, columns=["modalidad_trabajo"], dtype=int)

# Matriz de correlación
corr_ext = df_ext.corr()

fig_corr, ax_corr = plt.subplots(figsize=(9, 8))
imagen = ax_corr.imshow(corr_ext, cmap="coolwarm", vmin=-1, vmax=1)

ax_corr.set_xticks(range(len(corr_ext.columns)))
ax_corr.set_yticks(range(len(corr_ext.columns)))
ax_corr.set_xticklabels(corr_ext.columns, rotation=45, ha="right")
ax_corr.set_yticklabels(corr_ext.columns)

for i in range(len(corr_ext)):
    for j in range(len(corr_ext)):
        ax_corr.text(j, i, f"{corr_ext.iloc[i, j]:.2f}", ha="center", va="center")

fig_corr.colorbar(imagen, label="Coeficiente de correlación")
ax_corr.set_title("Matriz de correlación")
fig_corr.tight_layout()
plt.show()

X_ext = df_ext.drop(columns=["renuncia"])
y_ext = df_ext["renuncia"]

# División y parámetros iguales al escenario base
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_ext,
    y_ext,
    test_size=0.30,
    random_state=42,
    stratify=y_ext
)

resultados_ext = []

for nombre, modelo in modelos.items():

    modelo_e = clone(modelo)
    modelo_e.fit(X_train_e, y_train_e)
    y_pred_e = modelo_e.predict(X_test_e)

    cv_e = cross_val_score(modelo_e, X_ext, y_ext, cv=5, scoring="accuracy")

    resultados_ext.append({
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_test_e, y_pred_e),
        "Precision": precision_score(y_test_e, y_pred_e, zero_division=0),
        "Recall": recall_score(y_test_e, y_pred_e, zero_division=0),
        "F1-score": f1_score(y_test_e, y_pred_e, zero_division=0),
        "Validación cruzada": cv_e.mean()
    })

df_resultados_ext = pd.DataFrame(resultados_ext)

print("\n" + "="*70)
print("TABLA COMPARATIVA - ESCENARIO AMPLIADO (modalidad_trabajo + liderazgo)")
print("="*70)

print(df_resultados_ext.sort_values(by="F1-score", ascending=False))

# Gráfica: escenario base vs ampliado
metricas = ["Accuracy", "Precision", "Recall", "F1-score", "Validación cruzada"]
fig_cmp, ejes_cmp = plt.subplots(2, 3, figsize=(15, 8))
ejes_cmp = ejes_cmp.ravel()

for k, metrica in enumerate(metricas):
    pos = range(len(df_resultados))
    ejes_cmp[k].bar([p - 0.2 for p in pos], df_resultados[metrica],
                    width=0.4, label="Base (5 variables)")
    ejes_cmp[k].bar([p + 0.2 for p in pos], df_resultados_ext[metrica],
                    width=0.4, label="Ampliado (+ liderazgo, + modalidad)")
    ejes_cmp[k].set_xticks(list(pos))
    ejes_cmp[k].set_xticklabels(df_resultados["Modelo"], rotation=25)
    ejes_cmp[k].set_ylim(0, 1.1)
    ejes_cmp[k].set_title(metrica)

ejes_cmp[5].axis("off")
handles, labels = ejes_cmp[0].get_legend_handles_labels()
ejes_cmp[5].legend(handles, labels, loc="center")
fig_cmp.suptitle("Efecto de agregar liderazgo y modalidad de trabajo")
fig_cmp.tight_layout()
plt.show()