# Perceptrón - Clasificación del Dataset Iris

Este proyecto implementa el algoritmo del perceptrón desde cero para la clasificación binaria del famoso dataset Iris. Incluye múltiples experimentos que demuestran el comportamiento del perceptrón en diferentes escenarios, desde datos linealmente separables hasta casos no separables.

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Uso](#-uso)
- [Experimentos](#-experimentos)
- [Resultados](#-resultados)
- [Contribución](#-contribución)

## 🌟 Características

- **Implementación completa del perceptrón** desde cero
- **Múltiples experimentos** con diferentes configuraciones
- **Visualización interactiva** de regiones de decisión
- **Análisis de convergencia** y comportamiento del algoritmo
- **Comparación de casos separables vs no separables**
- **Estudio del impacto de diferentes learning rates**

## 📦 Requisitos

### Software necesario:
- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Dependencias de Python:
```
numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.3.0
scipy>=1.7.0
urllib3>=1.26.0
```

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/pedroam-dev/perceptron.git
cd perceptron
```

### 2. Crear un entorno virtual (recomendado)
```bash
# Con venv
python -m venv perceptron_env
source perceptron_env/bin/activate  # En macOS/Linux
# perceptron_env\Scripts\activate   # En Windows

# O con conda
conda create -n perceptron_env python=3.9
conda activate perceptron_env
```

### 3. Instalar dependencias
```bash
pip install numpy pandas matplotlib scipy
```

### 4. Verificar instalación
```bash
python -c "import numpy, pandas, matplotlib, scipy; print('¡Instalación exitosa!')"
```

## 📁 Estructura del Proyecto

```
perceptron/
│
├── README.md                    # Este archivo
├── iris.data                   # Dataset Iris (descarga automática)
├── iris.names.txt              # Descripción del dataset
│
├── perceptron.py               # Implementación base del perceptrón
├── perceptron_cero.py          # Perceptrón con inicialización en ceros
│
├── load_and_plot_data.py       # Carga y visualización de datos
├── training.py                 # Entrenamiento básico
├── training_mod_v1.py          # Entrenamiento con pesos en cero
├── training_mod_v2.py          # Entrenamiento con diferentes épocas
├── training_mod_v3.py          # Análisis de learning rates
└── training_no_lineal.py       # Caso no linealmente separable
```

## 🎯 Uso

### Experimento 1: Carga y Visualización de Datos
```bash
python load_and_plot_data.py
```
**Propósito:** Cargar el dataset Iris y visualizar la separabilidad de las clases Setosa vs Versicolor usando las características "sepal length" y "petal length".

**Salida esperada:**
- Gráfico de dispersión mostrando las dos clases
- Verificación de separabilidad lineal

---

### Experimento 2: Entrenamiento Básico
```bash
python training.py
```
**Propósito:** Entrenar el perceptrón básico con diferentes números de épocas (50, 100, 150).

**Salida esperada:**
- Gráficas de convergencia
- Visualización de regiones de decisión
- Análisis de precisión

---

### Experimento 3: Inicialización con Ceros
```bash
python training_mod_v1.py
```
**Propósito:** Comparar el comportamiento cuando los pesos se inicializan en cero en lugar de valores aleatorios.

**Parámetros:**
- Épocas: 50, 100, 200
- Inicialización: w₁=0, w₂=0, bias=0

---

### Experimento 4: Análisis de Learning Rates
```bash
python training_mod_v3.py
```
**Propósito:** Estudiar cómo diferentes learning rates [0.2, 0.4, 0.6, 0.8, 1.0] afectan la convergencia.

**Métricas analizadas:**
- Velocidad de convergencia
- Estabilidad del entrenamiento
- Precisión final
- Tasa de error (1 - accuracy)

---

### Experimento 5: Datos No Separables Linealmente
```bash
python training_no_lineal.py
```
**Propósito:** Demostrar las limitaciones del perceptrón con datos no separables usando Versicolor vs Virginica.

**Características utilizadas:**
- Sepal Width vs Petal Width
- Épocas: 50, 100, 150

**Características especiales:**
- Visualización de puntos mal clasificados
- Línea de decisión visible
- Análisis de superposición entre clases

## 🧪 Experimentos

### 📊 Resumen de Experimentos

| Experimento | Archivo | Clases | Características | Separable | Objetivo |
|-------------|---------|--------|----------------|-----------|----------|
| 1 | `load_and_plot_data.py` | Setosa vs Versicolor | Sepal Length, Petal Length | ✅ Sí | Visualización |
| 2 | `training.py` | Setosa vs Versicolor | Sepal Length, Petal Length | ✅ Sí | Entrenamiento básico |
| 3 | `training_mod_v1.py` | Setosa vs Versicolor | Sepal Length, Petal Length | ✅ Sí | Inicialización ceros |
| 4 | `training_mod_v3.py` | Setosa vs Versicolor | Sepal Length, Petal Length | ✅ Sí | Learning rates |
| 5 | `training_no_lineal.py` | Versicolor vs Virginica | Sepal Width, Petal Width | ❌ No | Limitaciones |

### 🎛️ Parámetros Configurables

Para modificar los parámetros de entrenamiento, edita las siguientes variables en cada archivo:

```python
# Learning rate
eta = 0.1

# Número de épocas  
n_iter = 50

# Semilla aleatoria
random_state = 1
```

## 📈 Resultados

### Casos Linealmente Separables
- **Convergencia:** Típicamente en menos de 10 épocas
- **Precisión:** 100% (clasificación perfecta)
- **Errores finales:** 0 actualizaciones por época

### Casos No Separables
- **Convergencia:** No alcanza 0 errores
- **Precisión:** ~70-85% (limitada por superposición)
- **Errores finales:** Persisten actualizaciones por época

### Impacto del Learning Rate
- **LR bajo (0.2-0.4):** Convergencia lenta pero estable
- **LR medio (0.6-0.8):** Balance óptimo
- **LR alto (1.0):** Convergencia rápida, posibles oscilaciones

## 🔧 Personalización

### Cambiar Dataset
Para usar tus propios datos, modifica la sección de carga en cualquier archivo:

```python
# Reemplazar esta sección
X = df.iloc[0:100, [0, 2]].values  # tus características
y = np.where(y == 'clase1', 0, 1)  # tus etiquetas
```

### Agregar Nuevas Características
```python
# Ejemplo: usar todas las características
X = df.iloc[0:100, [0, 1, 2, 3]].values
```

### Modificar Visualización
```python
# Cambiar colores y marcadores
colors = ('green', 'orange', 'purple')
markers = ('x', '+', 'D')
```

## 🐛 Solución de Problemas

### Error: "No module named 'numpy'"
```bash
pip install numpy pandas matplotlib scipy
```

### Error: "HTTP Error" al descargar datos
El código automáticamente usará el archivo local `iris.data` si no puede descargar desde la URL.

### Error: "X is not defined"
Asegúrate de ejecutar la sección de carga de datos antes del entrenamiento.

### Gráficas no se muestran
```bash
# En sistemas sin interfaz gráfica
pip install matplotlib
# Agregar al código:
import matplotlib
matplotlib.use('Agg')  # Para guardar sin mostrar
```

## 🤝 Contribución

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📚 Referencias

- Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain.
- Dataset Iris: Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems.
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/iris

## 👤 Autor

**Pedro AM** - [@pedroam-dev](https://github.com/pedroam-dev)

---

⭐ Si este proyecto te fue útil, ¡no olvides darle una estrella!