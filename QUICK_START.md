# Guía de Ejecución Rápida - Proyecto Perceptrón

## ⚡ Instalación Express

### Opción 1: Configuración Automática
```bash
# Clonar y configurar automáticamente
git clone https://github.com/pedroam-dev/perceptron.git
cd perceptron
bash setup.sh
```

### Opción 2: Instalación Manual
```bash
# Clonar repositorio
git clone https://github.com/pedroam-dev/perceptron.git
cd perceptron

# Instalar dependencias
pip install -r requirements.txt
```

## 🎯 Orden de Ejecución Recomendado

### 1. Exploración inicial (5 minutos)
```bash
python load_and_plot_data.py
```
**Qué hace:** Carga el dataset Iris y muestra si las clases son separables
**Resultado:** Gráfico de dispersión de Setosa vs Versicolor

### 2. Entrenamiento básico (3 minutos)
```bash
python training.py
```
**Qué hace:** Entrena perceptrón básico con 10 épocas
**Resultado:** Gráficas de convergencia y regiones de decisión

### 3. Comparación de épocas (5 minutos)
```bash
python training_mod_v2.py
```
**Qué hace:** Compara entrenamiento con 50, 100, 150 épocas
**Resultado:** Análisis comparativo de convergencia

### 4. Análisis de learning rates (7 minutos)
```bash
python training_mod_v3.py
```
**Qué hace:** Prueba diferentes learning rates [0.2, 0.4, 0.6, 0.8, 1.0]
**Resultado:** 4 gráficas comparativas del impacto del learning rate

### 5. Caso no separable (8 minutos)
```bash
python training_no_lineal.py
```
**Qué hace:** Demuestra limitaciones con datos no separables
**Resultado:** Visualización de superposición y líneas de decisión

## 🔧 Personalización Rápida

### Cambiar número de épocas:
Editar en cualquier archivo:
```python
n_iter = 100  # Cambiar este valor
```

### Cambiar learning rate:
```python
eta = 0.1  # Cambiar este valor
```

### Cambiar características:
```python
# Actual: sepal length (0) y petal length (2)
X = df.iloc[0:100, [0, 2]].values

# Cambiar a: sepal width (1) y petal width (3)
X = df.iloc[0:100, [1, 3]].values
```

## 📊 Resultados Esperados

### ✅ Casos Separables (Setosa vs Versicolor):
- Convergencia: ~5-10 épocas
- Precisión: 100%
- Errores finales: 0

### ❌ Casos No Separables (Versicolor vs Virginica):
- Convergencia: No alcanza 0 errores
- Precisión: ~75-85%
- Errores finales: 5-15

## 🚨 Solución de Problemas

### "ModuleNotFoundError"
```bash
pip install numpy pandas matplotlib scipy
```

### "NameError: name 'X' is not defined"
Ejecutar primero la sección de carga de datos en el archivo

### Gráficas no aparecen
Verificar que tienes interfaz gráfica o usar:
```python
plt.savefig('resultado.png')  # En lugar de plt.show()
```

## ⏱️ Tiempo Total Estimado
- **Instalación:** 5 minutos
- **Experimentos completos:** 25-30 minutos
- **Análisis de resultados:** 10-15 minutos
- **Total:** ~45 minutos para experiencia completa