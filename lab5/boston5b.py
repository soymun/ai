import os

import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from keras.api.models import Sequential
from keras.api.layers import Dense
from matplotlib import pyplot as plt

# URL для загрузки данных
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

# Загрузка данных
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
df = pd.read_csv(url, sep='\s+', header=None, names=column_names)

X = df[['RM', 'LSTAT']].values
y = df['MEDV'].values

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

# Визуализация
plt.scatter(y, predictions_interaction)
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.title('Линейная регрессия со взаимодействием факторов')
plt.show()