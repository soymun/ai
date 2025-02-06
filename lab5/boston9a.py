import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 10, 100)
y = 2 * X + 3 * X**2 + 1 + np.random.normal(0, 1, 100)

# Визуализация
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Нелинейная зависимость от 1 переменной')
plt.show()