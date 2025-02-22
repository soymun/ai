import numpy as np
import matplotlib.pyplot as plt

def wng(x, snr):  # Функция для генерации белого шума
    snr = 10**(snr / 10.0)
    xpower = np.sum(x**2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

# Генерация данных
np.random.seed()  # Установка seed для воспроизводимости результатов
X1 = np.linspace(0, 10, 100)  # Создание массива из 100 точек в диапазоне от 0 до 10 для переменной X1
X2 = np.linspace(0, 5, 100)   # Создание массива из 100 точек в диапазоне от 0 до 5 для переменной X2

# Генерация целевой переменной y с учетом взаимодействия факторов
# y = 2*X1 + 3*X2 + 4*X1*X2 + 1 + шум
# Взаимодействие факторов учитывается через произведение X1 * X2
y = 2 * X1 + 3 * X2 + 4 * X1 * X2 + 1

y_normal = y + np.random.normal(0, 1, 100)

# Добавление белого шума
snr = 10  # Отношение сигнал/шум в dB
white_noise = wng(y, snr)  # Генерация белого шума
y_with_white_noise = y + white_noise  # Добавление белого шума к данным

# Визуализация данных
fig = plt.figure(figsize=(14, 6))

# График с нормальным шумом
ax1 = fig.add_subplot(121, projection='3d')  # Первый подграфик
ax1.scatter(X1, X2, y_normal)
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('y')
ax1.set_title('Зависимость с нормальным шумом')

# График с белым шумом
ax2 = fig.add_subplot(122, projection='3d')  # Второй подграфик
ax2.scatter(X1, X2, y_with_white_noise)
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('y')
ax2.set_title('Зависимость с белым шумом')

plt.tight_layout()
plt.show()