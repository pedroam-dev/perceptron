### Training the perceptron model with non-linearly separable data
# X = features
# y = labels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from urllib.error import HTTPError
from perceptron import Perceptron


# ### A function for plotting decision regions
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')

def calculate_accuracy(classifier, X, y):
    """Calcular la precisión del clasificador"""
    predictions = classifier.predict(X)
    return np.mean(predictions == y)

# Load Data Iris      
try:
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    print('From URL:', s)
    df = pd.read_csv(s, header=None, encoding='utf-8')
except HTTPError:
    s = 'iris.data'
    print('From local Iris path:', s)
    df = pd.read_csv(s, header=None, encoding='utf-8')

print("Dataset Iris cargado exitosamente")
print(f"Forma del dataset: {df.shape}")

# === CASO NO SEPARABLE LINEALMENTE ===
# Opción 1: Versicolor vs Virginica (más difíciles de separar)
print(f"\n=== CASO NO SEPARABLE LINEALMENTE ===")
print("Usando Iris-versicolor vs Iris-virginica")
print("Características: Sepal Width vs Petal Width")

# Seleccionar versicolor (50-99) y virginica (100-149)
y = df.iloc[50:150, 4].values
y = np.where(y == 'Iris-versicolor', 0, 1)

# Extraer sepal width y petal width (columnas 1 y 3)
X = df.iloc[50:150, [1, 3]].values

print(f"Datos: {X.shape[0]} muestras")
print(f"Versicolor (0): {np.sum(y == 0)} muestras")
print(f"Virginica (1): {np.sum(y == 1)} muestras")

# Visualizar los datos primero
plt.figure(figsize=(10, 6))
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', 
           label='Versicolor', alpha=0.8, edgecolors='black', s=100)
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', 
           label='Virginica', alpha=0.8, edgecolors='black', s=100)
plt.xlabel('Sepal Width [cm]')
plt.ylabel('Petal Width [cm]')
plt.title('Iris Dataset: Versicolor vs Virginica\n(Sepal Width vs Petal Width) - NO SEPARABLE LINEALMENTE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Entrenar con diferentes números de épocas
epochs_list = [50, 100, 150]
results = {}

print(f"\n=== ENTRENAMIENTO CON DATOS NO SEPARABLES ===")

for n_epochs in epochs_list:
    print(f"\n--- Entrenando con {n_epochs} épocas ---")
    
    ppn = Perceptron(eta=0.1, n_iter=n_epochs, random_state=1)
    ppn.fit(X, y)
    
    # Calcular precisión final
    final_accuracy = calculate_accuracy(ppn, X, y)
    final_error_rate = 1 - final_accuracy
    
    results[n_epochs] = {
        'perceptron': ppn,
        'errors': ppn.errors_,
        'accuracy': final_accuracy,
        'error_rate': final_error_rate
    }
    
    print(f"Precisión final: {final_accuracy:.4f}")
    print(f"Tasa de error final: {final_error_rate:.4f}")
    print(f"Actualizaciones en última época: {ppn.errors_[-1]}")
    
    # Verificar si convergió
    if ppn.errors_[-1] == 0:
        print("✓ Convergió (0 errores)")
    else:
        print("✗ No convergió completamente")

# === GRÁFICAS COMPARATIVAS ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Perceptrón en Datos NO Separables Linealmente:\nVersicolor vs Virginica (Sepal Width vs Petal Width)', 
             fontsize=16)

for i, n_epochs in enumerate(epochs_list):
    ppn = results[n_epochs]['perceptron']
    
    # Gráfica de errores por época (fila superior)
    axes[0, i].plot(range(1, len(ppn.errors_) + 1), ppn.errors_, 
                    marker='o', linewidth=2, markersize=6, color='red')
    axes[0, i].set_xlabel('Épocas', fontsize=12)
    axes[0, i].set_ylabel('Número de actualizaciones', fontsize=12)
    axes[0, i].set_title(f'Errores por Época - {n_epochs} Épocas', fontsize=14)
    axes[0, i].grid(True, alpha=0.3)
    
    # Agregar línea horizontal en 0 para mostrar convergencia ideal
    axes[0, i].axhline(y=0, color='green', linestyle='--', alpha=0.7, 
                      label='Convergencia ideal')
    axes[0, i].legend()
    
    # Mostrar estadísticas en la gráfica
    final_errors = ppn.errors_[-1]
    accuracy = results[n_epochs]['accuracy']
    axes[0, i].text(0.05, 0.95, f'Errores finales: {final_errors}\nPrecisión: {accuracy:.3f}', 
                   transform=axes[0, i].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Gráfica de regiones de decisión (fila inferior)
    plt.sca(axes[1, i])
    plot_decision_regions(X, y, classifier=ppn)
    axes[1, i].set_xlabel('Sepal Width [cm]', fontsize=12)
    axes[1, i].set_ylabel('Petal Width [cm]', fontsize=12)
    axes[1, i].set_title(f'Función de Decisión - {n_epochs} Épocas', fontsize=14)
    axes[1, i].legend(loc='upper left')

plt.tight_layout()
plt.show()

# === ANÁLISIS COMPARATIVO ===
print(f"\n=== ANÁLISIS DE RESULTADOS ===")
print("Épocas\tPrecisión\tError Rate\tErrores Finales\tConvergió")
print("-" * 65)

for epochs in epochs_list:
    result = results[epochs]
    accuracy = result['accuracy']
    error_rate = result['error_rate'] 
    final_errors = result['errors'][-1]
    converged = "Sí" if final_errors == 0 else "No"
    
    print(f"{epochs}\t{accuracy:.4f}\t\t{error_rate:.4f}\t\t{final_errors}\t\t{converged}")

# Comparación con caso separable (opcional)
print(f"\n=== COMPARACIÓN CON CASO SEPARABLE ===")
print("Para comparar, entrenando con Setosa vs Versicolor (separable):")

# Datos separables para comparación
y_separable = df.iloc[0:100, 4].values
y_separable = np.where(y_separable == 'Iris-setosa', 0, 1)
X_separable = df.iloc[0:100, [0, 2]].values

ppn_separable = Perceptron(eta=0.1, n_iter=50, random_state=1)
ppn_separable.fit(X_separable, y_separable)
acc_separable = calculate_accuracy(ppn_separable, X_separable, y_separable)

print(f"Caso separable (Setosa vs Versicolor): Precisión = {acc_separable:.4f}")
print(f"Caso NO separable (Versicolor vs Virginica): Precisión = {results[50]['accuracy']:.4f}")
print(f"Diferencia: {acc_separable - results[50]['accuracy']:.4f}")