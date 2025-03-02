import os

import numpy as np
from sklearn.model_selection import train_test_split

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.api.models import Sequential
from keras.api.layers import Dense
import pandas as pd
from matplotlib import pyplot as plt

# Загрузка данных
# Используем URL для загрузки набора данных о ценах на жилье в Бостоне
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

# Названия столбцов для DataFrame
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

# Загрузка данных в DataFrame
# sep='\s+' - разделитель (пробелы или табуляции)
# header=None - отсутствие заголовка в файле
# names=column_names - задаем имена столбцов
df = pd.read_csv(url, sep='\s+', header=None, names=column_names)

# Разделение данных на обучающую и тестовую выборки
# test_size - доля тестовой выборки (например, 0.2 для 20%)
# random_state - для воспроизводимости результатов
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Выбор двух наиболее значимых признаков
# 'RM' - среднее количество комнат в доме
# 'LSTAT' - процент населения с низким статусом
# .values - преобразование в массив NumPy
X = train_df[['RM', 'LSTAT']].values

# Целевая переменная (MEDV - медианная стоимость дома в тысячах долларов)
y = train_df['MEDV'].values

X_test = test_df[['RM', 'LSTAT']].values

# Целевая переменная (MEDV - медианная стоимость дома в тысячах долларов)
y_test = test_df['MEDV'].values

# Создание модели
model = Sequential()  # Инициализация последовательной модели

# Добавление полносвязного слоя (Dense)
# 1 - количество нейронов (один выход, так как это регрессия)
# input_dim=2 - количество входных признаков (RM и LSTAT)
# activation='linear' - линейная функция активации (по умолчанию для регрессии)
model.add(Dense(1, input_dim=2, activation='linear'))

# Компиляция модели
# optimizer='adam' - алгоритм оптимизации Adam (адаптивный метод градиентного спуска)
# loss='mean_squared_error' - функция потерь (среднеквадратичная ошибка, MSE)
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
# X - входные данные (признаки)
# y - целевые значения (цена дома)
# epochs=100 - количество эпох обучения (полных проходов по данным)
# batch_size=10 - количество образцов, обрабатываемых за один шаг обновления весов
# verbose=1 - вывод информации о процессе обучения
model.fit(X, y, epochs=30, batch_size=10, validation_split=0.2)

# Предсказание
# model.predict(X) возвращает предсказанные значения для входных данных X

X_new1 = np.array([[6.6, 5], [7, 4]])
X_new_df = pd.DataFrame(X_new1, columns=['RM', 'LSTAT'])

predictions = model.predict(X_new_df)

print(predictions)
