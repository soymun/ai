import os
# Отключаем оптимизации OneDNN для совместимости
# OneDNN — это библиотека для оптимизации вычислений, но иногда она может вызывать проблемы с совместимостью
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Импорт необходимых модулей из Keras
from keras import layers  # Импорт слоев для построения модели
from keras import models  # Импорт модуля для создания моделей

# Создание последовательной модели (Sequential)
# Sequential — это линейный стек слоев, где каждый слой имеет ровно один входной и один выходной тензор
model = models.Sequential()

# Добавление сверточного слоя (Conv2D)
# 32 — количество фильтров (выходных каналов)
# (3, 3) — размер ядра свертки (3x3)
# activation='relu' — функция активации ReLU (Rectified Linear Unit)
# input_shape=(150, 150, 3) — форма входных данных: изображение 150x150 пикселей с 3 каналами (RGB)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

# Добавление слоя подвыборки (MaxPooling2D)
# (2, 2) — размер окна пулинга (2x2)
# MaxPooling уменьшает размерность изображения, сохраняя наиболее важные признаки
model.add(layers.MaxPooling2D((2, 2)))

# Добавление еще одного сверточного слоя
# 64 — количество фильтров
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Добавление слоя подвыборки
model.add(layers.MaxPooling2D((2, 2)))

# Добавление третьего сверточного слоя
# 128 — количество фильтров
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# Добавление слоя подвыборки
model.add(layers.MaxPooling2D((2, 2)))

# Добавление четвертого сверточного слоя
# 128 — количество фильтров
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# Добавление слоя подвыборки
model.add(layers.MaxPooling2D((2, 2)))

# Добавление слоя Flatten
# Flatten преобразует многомерный тензор в одномерный (выравнивает данные)
# Например, из формы (batch_size, height, width, channels) в (batch_size, height * width * channels)
model.add(layers.Flatten())

# Добавление полносвязного слоя (Dense)
# 512 — количество нейронов
# activation='relu' — функция активации ReLU
model.add(layers.Dense(512, activation='relu'))

# Добавление выходного слоя
# 1 — количество нейронов (для бинарной классификации)
# activation='sigmoid' — функция активации Sigmoid (для получения вероятности)
model.add(layers.Dense(1, activation='sigmoid'))

# Вывод структуры модели
# model.summary() показывает архитектуру модели, количество параметров и форму выходных данных на каждом слое
model.summary()

# Загрузка изображения
img_path = 'img.png'  # Указываем путь к изображению (фото кошки)

# Импорт утилит Keras для работы с изображениями
import keras.api.utils
# Импорт библиотеки numpy для работы с массивами
import numpy as np

# Загрузка изображения и изменение его размера до 150x150 пикселей
img = keras.utils.load_img(img_path, target_size=(150, 150))

# Преобразование изображения в массив numpy
img_tensor = keras.utils.img_to_array(img)

# Добавление дополнительной оси (batch dimension)
# Модель ожидает входные данные в формате (batch_size, height, width, channels)
img_tensor = np.expand_dims(img_tensor, axis=0)

# Нормализация изображения (приведение значений пикселей к диапазону [0, 1])
img_tensor /= 255.0

# Вывод формы тензора изображения
# Ожидаемая форма: (1, 150, 150, 3) — 1 изображение, 150x150 пикселей, 3 канала (RGB)
print("Форма тензора изображения:", img_tensor.shape)