# Predicción de Rotación de Empleados con Machine Learning

Caso de estudio de clasificación supervisada aplicado a Recursos Humanos.
Especialización en Inteligencia Artificial, Universidad de Cundinamarca.
Asignatura: Introducción a Machine Learning (2026-2).

**Autor:** Juan Manuel Pastor Salamanca
**Docente:** Ing. Wilson Mauro Rojas Reales

## Descripción

Se construye y compara un conjunto de modelos de clasificación para predecir
si un empleado renunciará a la empresa (`1`) o permanecerá (`0`), con el fin de
anticipar costos de contratación, capacitación y pérdida de conocimiento.

## Variables

| Variable | Descripción |
|---|---|
| `edad` | Edad del empleado |
| `salario` | Salario mensual |
| `años_empresa` | Años de permanencia en la empresa |
| `satisfaccion` | Nivel de satisfacción (1 a 10) |
| `horas_extra` | Horas extras promedio |
| `renuncia` | **Variable objetivo**: 0 = permanece, 1 = renuncia |

En el escenario ampliado se agregan `liderazgo` y `modalidad_trabajo`.
**Estas dos variables son simuladas con fines didácticos.**

## Modelos comparados

Regresión Logística, Árbol de Decisión, Random Forest, KNN, SVM (kernel lineal)
y Naive Bayes.

## Métricas

Accuracy, Precision, Recall, F1-score, matriz de confusión (TN, FP, FN, TP) y
validación cruzada.

## Instalación y ejecución

```bash
# 1. Clonar el repositorio
git clone https://github.com/[usuario]/[repositorio].git
cd [repositorio]

# 2. Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar
python C2_CasoRRHH.py
```

El script imprime las métricas en consola y abre las gráficas (matriz de
correlación, matrices de confusión, comparación de métricas, comparación entre
escenarios, y matriz de correlación aumentada). Cierra cada ventana para continuar con la ejecución.

## Resultados (escenario base)

| Modelo | F1 (test) | Validación cruzada |
|---|---|---|
| Regresión Logística | 1.00 | 1.00 |
| Árbol de Decisión | 1.00 | 1.00 |
| Random Forest | 1.00 | 1.00 |
| KNN | 1.00 | 1.00 |
| SVM | 1.00 | 0.95 |
| Naive Bayes | 0.67 | 0.95 |

Cinco modelos empatan en F1 en el conjunto de prueba.

## Consideraciones éticas

Este tipo de modelo debe usarse para apoyar la retención, con supervisión
humana, transparencia hacia los empleados y cumplimiento de la normativa de
protección de datos personales. No debe emplearse para sancionar o discriminar.

## Estructura

```
.
├── C1_CasoRRHH.py
├── requirements.txt
├── .gitignore
└── README.md
```
