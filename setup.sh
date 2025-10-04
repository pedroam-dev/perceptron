#!/bin/bash

# Script de configuración rápida para el proyecto Perceptrón
# Uso: bash setup.sh

echo "🚀 Configurando el proyecto Perceptrón..."

# Verificar si Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 no está instalado."
    echo "   Por favor instala Python 3.7 o superior desde https://python.org"
    exit 1
fi

echo "✅ Python detectado: $(python3 --version)"

# Crear entorno virtual
echo "📦 Creando entorno virtual..."
python3 -m venv perceptron_env

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source perceptron_env/bin/activate

# Actualizar pip
echo "📈 Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install -r requirements.txt

# Verificar instalación
echo "🧪 Verificando instalación..."
python -c "
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
print('✅ Todas las dependencias instaladas correctamente!')
print(f'   - NumPy: {np.__version__}')
print(f'   - Pandas: {pd.__version__}')
print(f'   - Matplotlib: {plt.matplotlib.__version__}')
print(f'   - SciPy: {scipy.__version__}')
"

echo ""
echo "🎉 ¡Configuración completada!"
echo ""
echo "Para activar el entorno virtual en el futuro:"
echo "   source perceptron_env/bin/activate"
echo ""
echo "Para ejecutar los experimentos:"
echo "   python load_and_plot_data.py"
echo "   python training.py"
echo "   python training_no_lineal.py"
echo ""
echo "Para desactivar el entorno virtual:"
echo "   deactivate"