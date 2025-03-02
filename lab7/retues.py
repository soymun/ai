import os

import numpy as np

# Отключаем оптимизации OneDNN для TensorFlow для совместимости
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Импорт необходимых модулей из Keras и других библиотек
from keras.api.datasets import reuters  # Набор данных Reuters
from keras.api.models import Sequential  # Последовательная модель нейронной сети
from keras.api.layers import Dense, Dropout  # Полносвязный слой и слой Dropout
from keras.api.utils import to_categorical  # Преобразование меток в one-hot encoding
from keras.src.legacy.preprocessing.text import Tokenizer  # Токенизатор для текста
import matplotlib.pyplot as plt  # Библиотека для визуализации данных


def print_predictions(model, x_test, y_test, num_samples=5, countUnits=32):
    # Выбираем случайные индексы из тестового набора
    random_indices = np.random.choice(len(x_test), num_samples, replace=False)

    # Получаем предсказания для выбранных данных
    predictions = model.predict(x_test[random_indices])

    # Преобразуем предсказания и фактические значения в метки классов
    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels = np.argmax(y_test[random_indices], axis=1)

    print(f"Модель с количеством нейронов - {countUnits}")
    # Выводим результаты
    for i in range(num_samples):
        print(f"Предсказанное значение - {predicted_labels[i]}")
        print(f"Фактическое значение - {actual_labels[i]}")
        print()

# Загрузка данных Reuters
# num_words=10000 — ограничиваем словарь 10,000 наиболее часто встречающихся слов
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

# Инициализация токенизатора с ограничением на 10,000 слов
tokenizer = Tokenizer(num_words=10000)

# Преобразование текстовых данных в бинарную матрицу (one-hot encoding)
# mode='binary' — каждая ячейка матрицы будет содержать 1, если слово присутствует, и 0 — если нет
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

# Преобразование меток в one-hot encoding
# num_classes=46 — количество классов (46 тем в наборе данных)
num_classes = 46
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Функция для создания модели нейронной сети
def create_model(units):
    model = Sequential()  # Инициализация последовательной модели
    # Добавление полносвязного слоя с ReLU-активацией и Dropout для предотвращения переобучения
    model.add(Dense(units, activation='relu', input_shape=(10000,)))  # Входной слой
    model.add(Dropout(0.5))  # Dropout с вероятностью 0.5
    model.add(Dense(units, activation='relu'))  # Скрытый слой
    model.add(Dropout(0.5))  # Dropout с вероятностью 0.5
    model.add(Dense(units, activation='relu'))  # Скрытый слой
    model.add(Dropout(0.5))  # Dropout с вероятностью 0.5
    model.add(Dense(num_classes, activation='softmax'))  # Выходной слой с softmax для многоклассовой классификации
    # Компиляция модели
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Список количества нейронов в слоях для тестирования
units_list = [32, 64, 128]
histories = []  # Список для хранения истории обучения каждой модели
models = []

# Обучение моделей с разным количеством нейронов
for units in units_list:
    model = create_model(units)  # Создание модели с заданным количеством нейронов
    # Обучение модели на тренировочных данных
    # epochs=10 — количество эпох обучения
    # batch_size=32 — размер батча
    # validation_split=0.2 — 20% данных используются для валидации
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    histories.append(history)  # Сохранение истории обучения
    models.append(model)
    # Оценка модели на тестовых данных
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'Model with {units} units - Test Loss: {loss}, Test Accuracy: {accuracy}')

# Визуализация результатов обучения
plt.figure(figsize=(18, 12))  # Создание фигуры для графиков

# Построение графиков для каждой модели
for i, units in enumerate(units_list):
    # График точности
    plt.subplot(3, 2, 2 * i + 1)  # Создание подграфика для точности
    plt.plot(histories[i].history['accuracy'], label='Тренировочная точность')  # Точность на тренировочных данных
    plt.plot(histories[i].history['val_accuracy'], label='Точность при валидации')  # Точность на валидационных данных
    plt.title(f'Модель с {units} units - Точность')  # Заголовок графика
    plt.xlabel('Эпохи')  # Подпись оси X
    plt.ylabel('Точность')  # Подпись оси Y
    plt.legend()  # Добавление легенды

    # График потерь
    plt.subplot(3, 2, 2 * i + 2)  # Создание подграфика для потерь
    plt.plot(histories[i].history['loss'], label='Тренировочные потери')  # Потери на тренировочных данных
    plt.plot(histories[i].history['val_loss'], label='Потери валидации')  # Потери на валидационных данных
    plt.title(f'Модель с  {units} units - Потери')  # Заголовок графика
    plt.xlabel('Эпохи')  # Подпись оси X
    plt.ylabel('Потери')  # Подпись оси Y
    plt.legend()  # Добавление легенды

    print_predictions(models[i], x_test, y_test, 5, units)

plt.tight_layout()  # Автоматическая настройка расположения графиков
plt.show()  # Отображение всех графиков