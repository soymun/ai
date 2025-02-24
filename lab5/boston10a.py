import os

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.api.models import Sequential
from keras.api.layers import Dense
import pandas as pd
from matplotlib import pyplot as plt

# Загрузка данных в DataFrame
# sep='\s+' - разделитель (пробелы или табуляции)
# header=None - отсутствие заголовка в файле
# names=column_names - задаем имена столбцов
df = pd.read_csv("data_normal1.csv", sep=',')

# Разделение данных на обучающую и тестовую выборки
# test_size - доля тестовой выборки (например, 0.2 для 20%)
# random_state - для воспроизводимости результатов
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
predictions = model.predict(X_test)

# Оценка качества моделей с помощью среднеквадратичной ошибки (MSE)
# Оценка линейной регрессии по двум факторам
mse_linear = mean_squared_error(y_test, predictions)
print("MSE линейной регрессии по двум факторам:", mse_linear)
# Визуализация
# Построение графика для сравнения фактических и предсказанных значений
plt.scatter(y_test, predictions)  # Точечный график
plt.xlabel('Фактические значения')  # Подпись оси X
plt.ylabel('Предсказанные значения')  # Подпись оси Y
plt.title('Линейная регрессия по двум факторам')  # Заголовок графика
plt.show()  # Отображение графика