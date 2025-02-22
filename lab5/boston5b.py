import os

import numpy as np
from sklearn.model_selection import train_test_split

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from keras.api.models import Sequential
from keras.api.layers import Dense
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
# 'RM' - среднее количество комнат в доме
# 'LSTAT' - процент населения с низким статусом
# .values - преобразование в массив NumPy
X = train_df[['RM', 'LSTAT']].values

# Целевая переменная (MEDV - медианная стоимость дома в тысячах долларов)
y = train_df['MEDV'].values

X_test = test_df[['RM', 'LSTAT']].values

# Целевая переменная (MEDV - медианная стоимость дома в тысячах долларов)
y_test = test_df['MEDV'].values

# Создание взаимодействия признаков
# X_interaction - новая матрица признаков, включающая:
# 1. Первый признак (RM)
# 2. Второй признак (LSTAT)
# 3. Произведение RM и LSTAT (взаимодействие признаков)
# np.column_stack объединяет эти три столбца в одну матрицу
X_interaction = np.column_stack((X[:, 0], X[:, 1], X[:, 0] * X[:, 1]))


# Создание взаимодействия признаков
# X_interaction - новая матрица признаков, включающая:
# 1. Первый признак (RM)
# 2. Второй признак (LSTAT)
# 3. Произведение RM и LSTAT (взаимодействие признаков)
# np.column_stack объединяет эти три столбца в одну матрицу
X_interaction_test = np.column_stack((X_test[:, 0], X_test[:, 1], X_test[:, 0] * X_test[:, 1]))

# Создание модели
model_interaction = Sequential()  # Инициализация последовательной модели

# Добавление полносвязного слоя (Dense)
# 1 - количество нейронов (один выход, так как это регрессия)
# input_dim=3 - количество входных признаков (RM, LSTAT и их взаимодействие)
# activation='linear' - линейная функция активации (по умолчанию для регрессии)
model_interaction.add(Dense(1, input_dim=3, activation='linear'))

# Компиляция модели
# optimizer='adam' - алгоритм оптимизации Adam (адаптивный метод градиентного спуска)
# loss='mean_squared_error' - функция потерь (среднеквадратичная ошибка, MSE)
model_interaction.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
# X_interaction - входные данные (признаки, включая взаимодействие)
# y - целевые значения (цена дома)
# epochs=100 - количество эпох обучения (полных проходов по данным)
# batch_size=10 - количество образцов, обрабатываемых за один шаг обновления весов
# verbose=1 - вывод информации о процессе обучения
model_interaction.fit(X_interaction, y, epochs=100, batch_size=10, validation_split=0.2)

# Предсказание
# model_interaction.predict(X_interaction) возвращает предсказанные значения для входных данных X_interaction
predictions_interaction = model_interaction.predict(X_interaction_test)

# Визуализация
plt.scatter(y_test, predictions_interaction)
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.title('Линейная регрессия со взаимодействием факторов')
plt.show()