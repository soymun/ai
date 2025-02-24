import os

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from keras.api.models import Sequential
from keras.api.layers import Dense
from matplotlib import pyplot as plt

# Загрузка данных в DataFrame
# sep='\s+' - разделитель (пробелы или табуляции)
# header=None - отсутствие заголовка в файле
# names=column_names - задаем имена столбцов
df = pd.read_csv("data_normal3.csv", sep=',')

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Выбор двух наиболее значимых признаков
# 'RM' - среднее количество комнат в доме
# 'LSTAT' - процент населения с низким статусом
# .values - преобразование в массив NumPy
X = train_df[['X1', 'X2']].values

# Целевая переменная (MEDV - медианная стоимость дома в тысячах долларов)
y = train_df['y'].values

X_test = test_df[['X1', 'X2']].values

# Целевая переменная (MEDV - медианная стоимость дома в тысячах долларов)
y_test = test_df['y'].values

# Создание нелинейных признаков
# X_nonlinear - новая матрица признаков, включающая:
# 1. Первый признак (RM)
# 2. Квадрат RM (RM**2)
# 3. Куб RM (RM**3)
# np.column_stack объединяет эти три столбца в одну матрицу
X_nonlinear = train_df['X1'].values
X_nonlinear = np.column_stack((X_nonlinear, X_nonlinear**2, X_nonlinear**3))

X_nonlinear_test = test_df['X1'].values
X_nonlinear_test = np.column_stack((X_nonlinear_test, X_nonlinear_test**2, X_nonlinear_test**3))

# Создание модели
model_nonlinear = Sequential()  # Инициализация последовательной модели

# Добавление полносвязного слоя (Dense)
# 1 - количество нейронов (один выход, так как это регрессия)
# input_dim=3 - количество входных признаков (RM, RM**2, RM**3)
# activation='linear' - линейная функция активации (по умолчанию для регрессии)
model_nonlinear.add(Dense(1, input_dim=3, activation='linear'))

# Компиляция модели
# optimizer='adam' - алгоритм оптимизации Adam (адаптивный метод градиентного спуска)
# loss='mean_squared_error' - функция потерь (среднеквадратичная ошибка, MSE)
model_nonlinear.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
# X_nonlinear - входные данные (признаки, включая нелинейные преобразования)
# y - целевые значения (цена дома)
# epochs=100 - количество эпох обучения (полных проходов по данным)
# batch_size=10 - количество образцов, обрабатываемых за один шаг обновления весов
# verbose=1 - вывод информации о процессе обучения
model_nonlinear.fit(X_nonlinear, y, epochs=30, batch_size=10, validation_split=0.2)

# Предсказание
# model_nonlinear.predict(X_nonlinear) возвращает предсказанные значения для входных данных X_nonlinear
predictions_nonlinear = model_nonlinear.predict(X_nonlinear_test)

# Оценка нелинейной модели
mse_nonlinear = mean_squared_error(y_test, predictions_nonlinear)
print("MSE нелинейной модели:", mse_nonlinear)

# Визуализация
# Построение графика для сравнения фактических и предсказанных значений
plt.scatter(y_test, predictions_nonlinear)  # Точечный график
plt.xlabel('Фактические значения')  # Подпись оси X
plt.ylabel('Предсказанные значения')  # Подпись оси Y
plt.title('Нелинейная модель от одного фактора')  # Заголовок графика
plt.show()  # Отображение графика
