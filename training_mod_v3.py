### Training the perceptron model with different learning rates
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

def calculate_error_rate(classifier, X, y):
    """Calcular la tasa de error (1 - accuracy)"""
    return 1 - calculate_accuracy(classifier, X, y)

# Load Data Iris      
try:
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    print('From URL:', s)
    df = pd.read_csv(s, header=None, encoding='utf-8')
except HTTPError:
    s = 'iris.data'
    print('From local Iris path:', s)
    df = pd.read_csv(s, header=None, encoding='utf-8')

# Seleccionar setosa y versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# Extraer sepal length y petal length
X = df.iloc[0:100, [0, 2]].values

print(f"Datos cargados: X shape = {X.shape}, y shape = {y.shape}")

# Parámetros de entrenamiento
learning_rates = [0.2, 0.4, 0.6, 0.8, 1.0]
n_epochs = 200  # Cambiar epocas a 50, 100 y 200 para observar convergencia

# Diccionario para almacenar resultados
results = {}

print(f"\n=== ENTRENAMIENTO CON DIFERENTES LEARNING RATES ===")
print(f"Épocas: {n_epochs}")
print(f"Learning rates: {learning_rates}")

# Entrenar con diferentes learning rates
for lr in learning_rates:
    print(f"\n--- Training con Learning Rate = {lr} ---")
    
    ppn = Perceptron(eta=lr, n_iter=n_epochs, random_state=1)
    ppn.fit(X, y)
    
    # Calcular error rate final
    final_accuracy = calculate_accuracy(ppn, X, y)
    final_error_rate = 1 - final_accuracy
    
    # Almacenar resultados
    results[lr] = {
        'perceptron': ppn,
        'errors_per_epoch': ppn.errors_,
        'final_accuracy': final_accuracy,
        'final_error_rate': final_error_rate
    }
    
    print(f"Precisión final: {final_accuracy:.4f}")
    print(f"Tasa de error final: {final_error_rate:.4f}")
    print(f"Actualizaciones en última época: {ppn.errors_[-1]}")

# === GRÁFICAS ===

# 1. Gráfica de número de actualizaciones por época
plt.figure(figsize=(15, 10))

# Subplot 1: Número de actualizaciones (errores) por época
plt.subplot(2, 2, 1)
for lr in learning_rates:
    plt.plot(range(1, len(results[lr]['errors_per_epoch']) + 1), 
             results[lr]['errors_per_epoch'], 
             marker='o', label=f'η = {lr}', linewidth=2, markersize=4)

plt.xlabel('Épocas')
plt.ylabel('Número de actualizaciones')
plt.title('Número de actualizaciones por época')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Tasa de error final para cada learning rate
plt.subplot(2, 2, 2)
error_rates = [results[lr]['final_error_rate'] for lr in learning_rates]
plt.bar([str(lr) for lr in learning_rates], error_rates, 
        color=['red', 'orange', 'yellow', 'lightgreen', 'blue'], alpha=0.7)
plt.xlabel('Learning Rate')
plt.ylabel('Tasa de error final (1 - Accuracy)')
plt.title('Tasa de error final vs Learning Rate')
plt.grid(True, alpha=0.3, axis='y')

# Agregar valores en las barras
for i, rate in enumerate(error_rates):
    plt.text(i, rate + 0.005, f'{rate:.3f}', ha='center', va='bottom')

# Subplot 3: Convergencia (épocas hasta llegar a 0 errores)
plt.subplot(2, 2, 3)
convergence_epochs = []
convergence_labels = []

for lr in learning_rates:
    errors = results[lr]['errors_per_epoch']
    converged_at = None
    for epoch, err in enumerate(errors):
        if err == 0:
            converged_at = epoch + 1
            break
    
    if converged_at:
        convergence_epochs.append(converged_at)
        convergence_labels.append(f'η = {lr}')
    else:
        convergence_epochs.append(n_epochs)  # No convergió
        convergence_labels.append(f'η = {lr} (NC)')

plt.bar(convergence_labels, convergence_epochs, 
        color=['red', 'orange', 'yellow', 'lightgreen', 'blue'], alpha=0.7)
plt.xlabel('Learning Rate')
plt.ylabel('Época de convergencia')
plt.title('Época de convergencia por learning Rate')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# Subplot 4: Comparación de precisión final
plt.subplot(2, 2, 4)
accuracies = [results[lr]['final_accuracy'] for lr in learning_rates]
plt.plot(learning_rates, accuracies, marker='o', linewidth=3, markersize=8, color='green')
plt.xlabel('Learning Rate')
plt.ylabel('Precisión final')
plt.title('Precisión final vs Learning Rate')
plt.grid(True, alpha=0.3)
plt.ylim(0.95, 1.01)

# Agregar valores en los puntos
for lr, acc in zip(learning_rates, accuracies):
    plt.annotate(f'{acc:.3f}', (lr, acc), textcoords="offset points", 
                xytext=(0,10), ha='center')

plt.tight_layout()
plt.show()

# === ANÁLISIS DETALLADO ===
print(f"\n=== ANÁLISIS DE RESULTADOS ===")
print("Learning Rate\tPrecisión\tError Rate\tConvergencia")
print("-" * 55)

for lr in learning_rates:
    ppn = results[lr]['perceptron']
    accuracy = results[lr]['final_accuracy']
    error_rate = results[lr]['final_error_rate']
    
    # Encontrar convergencia
    converged_at = "No convergió"
    for epoch, errors in enumerate(ppn.errors_):
        if errors == 0:
            converged_at = f"Época {epoch + 1}"
            break
    
    print(f"{lr}\t\t{accuracy:.4f}\t\t{error_rate:.4f}\t\t{converged_at}")

