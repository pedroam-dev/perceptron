### Training the perceptron model
# X = features
# y = labels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from urllib.error import HTTPError
from perceptron_cero import Perceptron


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
print("Inicialización de pesos: CEROS (modificado)")

# Entrenar con diferentes números de épocas (CAMBIADO: 50, 100, 200)
epochs_list = [50, 100, 200]

for n_epochs in epochs_list:
    print(f"\n--- Entrenando con {n_epochs} épocas ---")
    print(f"Inicialización: w1=0.0, w2=0.0, bias=0.0")
    
    ppn = Perceptron(eta=0.1, n_iter=n_epochs)
    ppn.fit(X, y)
    
    # Gráfica de convergencia
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.title(f'Convergencia - {n_epochs} épocas (pesos iniciales = 0)')
    plt.grid(True, alpha=0.3)
    
    # Encontrar punto de convergencia
    convergence_epoch = None
    for epoch, errors in enumerate(ppn.errors_):
        if errors == 0:
            convergence_epoch = epoch + 1
            break
    
    if convergence_epoch:
        plt.axvline(x=convergence_epoch, color='red', linestyle='--', 
                   label=f'Convergencia: época {convergence_epoch}')
        plt.legend()
    
    # Gráfica de regiones de decisión
    plt.subplot(1, 2, 2)
    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.title(f'Función de decisión - {n_epochs} épocas')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar estadísticas
    final_errors = ppn.errors_[-1]
    print(f"Errores finales: {final_errors}")
    print(f"Pesos finales: w1={ppn.w_[0]:.4f}, w2={ppn.w_[1]:.4f}, bias={ppn.b_:.4f}")
    
    if convergence_epoch:
        print(f"Convergió en la época: {convergence_epoch}")
    else:
        print("No convergió completamente")
