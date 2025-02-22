import numpy as np
import matplotlib.pyplot as plt

def wng(x, snr):  # Функция для генерации белого шума
    snr = 10**(snr / 10.0)
    xpower = np.sum(x**2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

# Генерация данных
np.random.seed()  # Установка seed для воспроизводимости результатов
X = np.linspace(0, 10, 100)  # Создание массива из 100 точек в диапазоне от 0 до 10
y = 2 * X + 1  # Генерация целевой переменной y = 2*X + 1 + нормальный шум


y_normal = y + np.random.normal(0, 1, 100)
# Добавление белого шума
snr = 10  # Отношение сигнал/шум в dB
white_noise = wng(y, snr)  # Генерация белого шума
y_with_white_noise = y + white_noise  # Добавление белого шума к данным

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