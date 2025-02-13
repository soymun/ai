import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Xn = 0
Xk = 6.4
deltaX = (Xk) / 3
N = 150
noise_level = 10

def white_noise(x, snr):
    snr = 10**(snr / 10.0)
    xpower = np.sum(x**2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

# Генерация данных для каждого кластера с шумом по X и Y
def generate_cluster_noise(Xn, Xk, N, formula, noise_level):
    x = np.linspace(Xn, Xk, N)
    y = formula(x) + white_noise(x, noise_level)
    x_noise = x + white_noise(formula(x), noise_level)  # Добавляем шум по X
    return x_noise, y

def generate_cluster(Xn, Xk, N, formula):
    x = np.linspace(Xn, Xk, N)
    y = formula(x)
    return x, y

def formula_1(x):
    return 1.5 + 0.1 * x**2

def formula_2(x):
    return 0.5 - 0.35 * x

def formula_3(x):
    return 1 + x**2 * (1 - 1.1 * np.sin(0.25 * x + 0.4)**2)

x1n, y1n = generate_cluster_noise(Xn, Xn + deltaX, N, formula_1, noise_level)
x2n, y2n = generate_cluster_noise(Xn + deltaX, Xn + (2 * deltaX), N, formula_2, noise_level)
x3n, y3n = generate_cluster_noise(Xn + (2 * deltaX), Xn + (3 * deltaX), N, formula_3, noise_level)

x1, y1 = generate_cluster(Xn, Xn + deltaX, N, formula_1)
x2, y2 = generate_cluster(Xn + deltaX, Xn + (2 * deltaX), N, formula_2)
x3, y3 = generate_cluster(Xn + (2 * deltaX), Xn + (3 * deltaX), N, formula_3)

x = np.concatenate((x1, x2, x3))
y = np.concatenate((y1, y2, y3))

df1 = pd.DataFrame({'X': x1n, 'Y': y1n, 'Target': 0})
df2 = pd.DataFrame({'X': x2n, 'Y': y2n, 'Target': 1})
df3 = pd.DataFrame({'X': x3n, 'Y': y3n, 'Target': 2})

df = pd.concat([df1, df2, df3], ignore_index=True)

df.to_csv('data_set.csv', index=False)

plt.figure(figsize=(10, 6))
plt.scatter(df1['X'], df1['Y'], label='Cluster 1', alpha=0.6)
plt.scatter(df2['X'], df2['Y'], label='Cluster 2', alpha=0.6)
plt.scatter(df3['X'], df3['Y'], label='Cluster 3', alpha=0.6)
plt.plot(x, y, label='Combined Line', color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Generated Clusters and Combined Line')
plt.show()

print(df.head())