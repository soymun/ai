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
X = np.linspace(0, 10, 100)  # Создание массива из 100 точек в диапазоне от 0 до 10
y = 2 * X + 1  # Генерация целевой переменной y = 2*X + 1 + нормальный шум


y_normal = y + np.random.normal(0, 10, 100)
# Добавление белого шума
snr = 10  # Отношение сигнал/шум в dB
white_noise = wng(y, snr)  # Генерация белого шума
y_with_white_noise = y + white_noise  # Добавление белого шума к данным

# Создание DataFrame для сохранения в CSV
data_normal = pd.DataFrame({'X': X, 'y': y_normal})  # Данные с нормальным шумом
data_white = pd.DataFrame({'X': X, 'y': y_with_white_noise})  # Данные с белым шумом

# Сохранение в CSV
data_normal.to_csv('data_normal1.csv', index=False)  # Файл с нормальным шумом
data_white.to_csv('data_white1.csv', index=False)  # Файл с белым шумом

# Визуализация данных
plt.figure(figsize=(12, 6))

# График с нормальным шумом
plt.subplot(1, 2, 1)
plt.scatter(X, y_normal)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Линейная зависимость с нормальным шумом')

# График с белым шумом
plt.subplot(1, 2, 2)
plt.scatter(X, y_with_white_noise)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Линейная зависимость с белым шумом')

plt.tight_layout()
plt.show()