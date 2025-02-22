import os

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.api.models import Sequential
from keras.api.layers import Dense
import pandas as pd
from matplotlib import pyplot as plt

# URL для загрузки данных
# Ссылка на набор данных о ценах на жилье в Бостоне, размещенный в репозитории UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

# Загрузка данных
# Список названий столбцов для DataFrame
column_names = [
    'CRIM',  # Уровень преступности на душу населения
    'ZN',    # Доля жилых участков, отведенных под дома с большими участками
    'INDUS', # Доля не розничных торговых площадей в городе
    'CHAS',  # Наличие реки (1 - есть, 0 - нет)
    'NOX',   # Концентрация оксидов азота
    'RM',    # Среднее количество комнат в доме
    'AGE',   # Доля домов, построенных до 1940 года
    'DIS',   # Взвешенное расстояние до пяти рабочих центров Бостона
    'RAD',   # Индекс доступности радиальных магистралей
    'TAX',   # Ставка налога на имущество
    'PTRATIO', # Соотношение учеников и учителей в школах
    'B',     # Доля афроамериканского населения
    'LSTAT', # Доля населения с низким статусом
    'MEDV'   # Медианная стоимость дома (целевая переменная)
]

# Загрузка данных в DataFrame
# sep='\s+' - разделитель (пробелы или табуляции)
# header=None - отсутствие заголовка в файле
# names=column_names - задаем имена столбцов
df = pd.read_csv(url, sep='\s+', header=None, names=column_names)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Выбор двух наиболее значимых признаков
# X - матрица признаков, содержащая два признака: 'RM' (среднее количество комнат) и 'LSTAT' (доля населения с низким статусом)
# y - целевая переменная 'MEDV' (медианная стоимость дома)
X = train_df[['RM', 'LSTAT']].values
y = train_df['MEDV'].values

X_TEST = test_df[['RM', 'LSTAT']].values
y_TEST = test_df['MEDV'].values

# Создание модели линейной регрессии по двум факторам
model = Sequential()  # Инициализация последовательной модели
model.add(Dense(1, input_dim=2, activation='linear'))  # Полносвязный слой с одним выходом и линейной активацией
model.compile(optimizer='adam', loss='mean_squared_error')  # Компиляция модели с оптимизатором Adam и функцией потерь MSE
model.fit(X, y, epochs=100, batch_size=10, validation_split=0.2)  # Обучение модели на 100 эпох с размером батча 10
predictions = model.predict(X_TEST)  # Предсказание на обучающих данных

# Создание взаимодействия признаков
# X_interaction - новая матрица признаков, включающая:
# 1. Первый признак (RM)
# 2. Второй признак (LSTAT)
# 3. Произведение RM и LSTAT (взаимодействие признаков)
X_interaction = np.column_stack((X[:, 0], X[:, 1], X[:, 0] * X[:, 1]))

X_interaction_TEST = np.column_stack((X_TEST[:, 0], X_TEST[:, 1], X_TEST[:, 0] * X_TEST[:, 1]))

# Создание модели линейной регрессии со взаимодействием факторов
model_interaction = Sequential()  # Инициализация последовательной модели
model_interaction.add(Dense(1, input_dim=3, activation='linear'))  # Полносвязный слой с одним выходом и линейной активацией
model_interaction.compile(optimizer='adam', loss='mean_squared_error')  # Компиляция модели с оптимизатором Adam и функцией потерь MSE
model_interaction.fit(X_interaction, y, epochs=100, batch_size=10, validation_split=0.2)  # Обучение модели на 100 эпох с размером батча 10
predictions_interaction = model_interaction.predict(X_interaction_TEST)  # Предсказание на обучающих данных

# Создание нелинейных признаков
# X_nonlinear - новая матрица признаков, включающая:
# 1. Первый признак (RM)
# 2. Квадрат RM (RM**2)
# 3. Куб RM (RM**3)
X_nonlinear = train_df['RM'].values
X_nonlinear = np.column_stack((X_nonlinear, X_nonlinear**2, X_nonlinear**3))

X_nonlinear_test = test_df['RM'].values
X_nonlinear_test = np.column_stack((X_nonlinear_test, X_nonlinear_test**2, X_nonlinear_test**3))

# Создание нелинейной модели
model_nonlinear = Sequential()  # Инициализация последовательной модели
model_nonlinear.add(Dense(1, input_dim=3, activation='linear'))  # Полносвязный слой с одним выходом и линейной активацией
model_nonlinear.compile(optimizer='adam', loss='mean_squared_error')  # Компиляция модели с оптимизатором Adam и функцией потерь MSE
model_nonlinear.fit(X_nonlinear, y, epochs=100, batch_size=10, validation_split=0.2)  # Обучение модели на 100 эпох с размером батча 10
predictions_nonlinear = model_nonlinear.predict(X_nonlinear_test)  # Предсказание на обучающих данных

# Оценка качества моделей с помощью среднеквадратичной ошибки (MSE)
# Оценка линейной регрессии по двум факторам
mse_linear = mean_squared_error(y_TEST, predictions)
print("MSE линейной регрессии по двум факторам:", mse_linear)

# Оценка линейной регрессии со взаимодействием факторов
mse_interaction = mean_squared_error(y_TEST, predictions_interaction)
print("MSE линейной регрессии со взаимодействием факторов:", mse_interaction)

# Оценка нелинейной модели
mse_nonlinear = mean_squared_error(y_TEST, predictions_nonlinear)
print("MSE нелинейной модели:", mse_nonlinear)