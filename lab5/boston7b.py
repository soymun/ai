import numpy as np
import matplotlib.pyplot as plt

# Генерация данных
X1 = np.linspace(0, 10, 100)
X2 = np.linspace(0, 5, 100)
y = 2 * X1 + 3 * X2 + 1 + np.random.normal(0, 1, 100)

# Визуализация
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, y)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
plt.title('Линейная зависимость с шумом от 2 переменных')
plt.show()