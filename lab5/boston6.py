import os

import numpy as np
from sklearn.metrics import mean_squared_error

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.api.models import Sequential
from keras.api.layers import Dense
import pandas as pd
from matplotlib import pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

# Загрузка данных
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
df = pd.read_csv(url, sep='\s+', header=None, names=column_names)

# Выбор двух наиболее значимых признаков
X = df[['RM', 'LSTAT']].values
y = df['MEDV'].values

# Создание модели
model = Sequential()
model.add(Dense(1, input_dim=2, activation='linear'))

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model.fit(X, y, epochs=100, batch_size=10, verbose=1)

# Предсказание
predictions = model.predict(X)

X_interaction = np.column_stack((X[:, 0], X[:, 1], X[:, 0] * X[:, 1]))

# Создание модели
model_interaction = Sequential()
model_interaction.add(Dense(1, input_dim=3, activation='linear'))

# Компиляция модели
model_interaction.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model_interaction.fit(X_interaction, y, epochs=100, batch_size=10, verbose=1)

# Предсказание
predictions_interaction = model_interaction.predict(X_interaction)

X_nonlinear = df['RM'].values
X_nonlinear = np.column_stack((X_nonlinear, X_nonlinear**2, X_nonlinear**3))

# Создание модели
model_nonlinear = Sequential()
model_nonlinear.add(Dense(1, input_dim=3, activation='linear'))

# Компиляция модели
model_nonlinear.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model_nonlinear.fit(X_nonlinear, y, epochs=100, batch_size=10, verbose=1)

# Предсказание
predictions_nonlinear = model_nonlinear.predict(X_nonlinear)

mse_linear = mean_squared_error(y, predictions)
print("MSE линейной регрессии по двум факторам:", mse_linear)

# Оценка линейной регрессии со взаимодействием факторов
mse_interaction = mean_squared_error(y, predictions_interaction)
print("MSE линейной регрессии со взаимодействием факторов:", mse_interaction)

# Оценка нелинейной модели
mse_nonlinear = mean_squared_error(y, predictions_nonlinear)
print("MSE нелинейной модели:", mse_nonlinear)