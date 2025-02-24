import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def wng(x, snr):  # Функция для генерации белого шума
    snr = 10**(snr / 10.0)
    xpower = np.sum(x**2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

# Генерация данных
np.random.seed()  # Установка seed для воспроизводимости результатов
X1 = np.linspace(0, 10, 1000)  # Создание массива из 100 точек в диапазоне от 0 до 10 для переменной X1
X2 = np.linspace(0, 5, 1000)   # Создание массива из 100 точек в диапазоне от 0 до 5 для переменной X2
y = 2 * X1 + 3 * X2 + 1  # Генерация целевой переменной y = 2*X1 + 3*X2 + 1 + нормальный шум

y_normal = y + np.random.normal(0, 10, 1000)

# Добавление белого шума
snr = 10  # Отношение сигнал/шум в dB
white_noise = wng(y, snr)  # Генерация белого шума
y_with_white_noise = y + white_noise  # Добавление белого шума к данным

# Создание DataFrame для сохранения в CSV
data_normal = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y_normal})  # Данные с нормальным шумом
data_white = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y_with_white_noise})  # Данные с белым шумом

# Сохранение в CSV
data_normal.to_csv('data_normal1.csv', index=False)  # Файл с нормальным шумом
data_white.to_csv('data_white1.csv', index=False)  # Файл с белым шумом


# Визуализация данных
fig = plt.figure(figsize=(14, 6))

# График с нормальным шумом
ax1 = fig.add_subplot(121, projection='3d')  # Первый подграфик
ax1.scatter(X1, X2, y_normal)
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('y')
ax1.set_title('Линейная зависимость с нормальным шумом')

# График с белым шумом
ax2 = fig.add_subplot(122, projection='3d')  # Второй подграфик
ax2.scatter(X1, X2, y_with_white_noise)
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('y')
ax2.set_title('Линейная зависимость с белым шумом')

plt.tight_layout()
plt.show()