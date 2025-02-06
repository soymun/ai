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

# Визуализация
plt.scatter(y, predictions_nonlinear)
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.title('Нелинейная модель от одного фактора')
plt.show()