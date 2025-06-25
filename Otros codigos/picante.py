import numpy as np
import scipy

# definir una función para la optimización
def func(x):
    return x**2 + 10*np.sin(x)

# encontrar el mínimo de la función
result = scipy.optimize.minimize(func, 0)

# imprimir el resultado de la optimización
print(result)