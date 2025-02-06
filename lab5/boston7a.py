import numpy as np
import matplotlib.pyplot as plt

# Генерация данных
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 1, 100)

# Визуализация
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Линейная зависимость с шумом от 1 переменной')
plt.show()