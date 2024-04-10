import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Coeficiente eficiencia
alpha = 1 # Pendiente de la eficiencia

# Coeficientes de delta Cl
beta = 40 # Alto de la campana
gamma = 15 # Ancho de la campana


# Definir los rangos de E (eficiencia) y |ΔCl| (desviación del Cl)
E = np.linspace(0, 50, 100)  # Eficiencia aerodinámica
delta_Cl = np.linspace(-2, 2, 50)  # Desviación de Cl

# Crear una malla de valores de E y |ΔCl|
E, delta_Cl = np.meshgrid(E, delta_Cl)

# Calcular la recompensa R basado en la función dada
R = alpha * E + beta * np.exp(-gamma * (delta_Cl)**2)



# Crear la figura y el eje para la gráfica 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie
surf = ax.plot_surface(E, delta_Cl, R, cmap='viridis', edgecolor='none')

# Añadir títulos y etiquetas
ax.set_title('Función de Recompensa en 3D')
ax.set_xlabel('Eficiencia (E)')
ax.set_ylabel('Desviación de $C_l$ (|Δ$C_l$|)')
ax.set_zlabel('Recompensa (R)')

# Añadir una barra de colores para indicar los valores de la recompensa
fig.colorbar(surf, shrink=0.5, aspect=5)

# Mostrar la gráfica
plt.show()