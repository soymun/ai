import numpy as np
import matplotlib.pyplot as plt

# Генерация данных
X1 = np.linspace(0, 10, 100)
X2 = np.linspace(0, 5, 100)
X3 = np.linspace(0, 2, 100)
y = 2 * X1 + 3 * X2 + 4 * X3 + 1 + np.random.normal(0, 1, 100)

# Визуализация (можно использовать 3D график для первых двух переменных и цвет для третьей)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X1, X2, y, c=X3, cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
plt.colorbar(scatter, label='X3')
plt.title('Линейная зависимость с шумом от 3 переменных')
plt.show()