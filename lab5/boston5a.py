import os

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

# Визуализация
plt.scatter(y, predictions)
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.title('Линейная регрессия по двум факторам')
plt.show()