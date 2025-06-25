import matplotlib.pyplot as plt 
import numpy as np

# crear un array de 100 números aleatorios
arr = np.random.rand(100)

# crear un gráfico de línea con los números aleatorios
plt.plot(arr)

# agregar títulos y etiquetas al gráfico
plt.title("Gráfico de línea")
plt.xlabel("X")
plt.ylabel("Y")

# mostrar el gráfico
plt.show()