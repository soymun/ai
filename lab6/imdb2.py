# %%
"""
# IMDB
"""
import os

# Отключаем оптимизации OneDNN для TensorFlow, чтобы избежать потенциальных проблем с совместимостью
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# %%
import matplotlib.pyplot as plt  # Импорт библиотеки для построения графиков
import numpy as np  # Импорт библиотеки для работы с массивами и математическими операциями
import keras  # Импорт библиотеки Keras для создания и обучения нейронных сетей
from keras.api import layers, optimizers, losses, metrics, activations  # Импорт необходимых модулей Keras

# Подготовка данных
# Функция для перекодирования последовательностей в бинарные векторы (0-1)
def vectorize(sequences):
    """
    Векторизация последовательностей (преобразование текста в бинарный вектор)

    :param sequences: Список последовательностей (индексов слов)
    :return: Массив бинарных векторов
    """
    results = np.zeros((len(sequences), NUM_WORDS))  # Создаем нулевой массив размером (количество последовательностей, NUM_WORDS)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1  # Устанавливаем 1 на позициях, соответствующих индексам слов в последовательности
    return results


# Параметры для обучения модели
NUM_WORDS = 10000  # Количество наиболее часто встречающихся слов, которые будут использоваться
INITIAL_LR = 0.001  # Начальная скорость обучения

# Загрузка датасета IMDB
(data, labels), (data2, labels2) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

# Объединение датасетов
data = np.concatenate((data, data2), axis=0)  # Объединяем данные по строкам
labels = np.concatenate((labels, labels2), axis=0)  # Объединяем метки по строкам

# Перемешивание данных
indices = np.arange(data.shape[0])  # Создаем массив индексов
np.random.shuffle(indices)  # Перемешиваем индексы

data = data[indices]  # Перемешиваем данные
labels = labels[indices]  # Перемешиваем метки

# Векторизация данных
x = vectorize(data)  # Преобразуем данные в бинарные векторы
y = np.asarray(labels).astype("float32")  # Преобразуем метки в массив float32

# Разделение данных на обучающий и тестовый наборы
# Первые 10000 отзывов - тестовый набор, остальные 40000 - обучающий
train_x, train_y, test_x, test_y = x[10000:], y[10000:], x[:10000], y[:10000]


# Функция для создания и обучения модели
def train(batch_size, epochs):
    """
    Создание и обучение модели нейронной сети

    :param batch_size: Размер пакета (количество образцов за один шаг обновления весов)
    :param epochs: Количество эпох обучения
    :return: История обучения и модель
    """
    # Создание модели
    model = keras.Sequential(
        [
            layers.Input(shape=(NUM_WORDS,)),  # Входной слой с размерностью NUM_WORDS
            layers.Dense(32, activation=activations.relu),  # Полносвязный слой с 32 нейронами и функцией активации ReLU
            layers.Dropout(0.3),  # Слой Dropout для предотвращения переобучения
            layers.Dense(16, activation=activations.relu),  # Полносвязный слой с 16 нейронами и функцией активации ReLU
            layers.Dropout(0.25),  # Слой Dropout
            layers.Dense(1, activation=activations.sigmoid),  # Выходной слой с 1 нейроном и функцией активации sigmoid
        ]
    )

    # Компиляция модели
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=INITIAL_LR),  # Оптимизатор RMSprop с заданной скоростью обучения
        loss=losses.BinaryCrossentropy(),  # Функция потерь для бинарной классификации
        metrics=[metrics.BinaryAccuracy()],  # Метрика - точность (accuracy)
    )

    # Обучение модели
    return (
        model.fit(
            train_x,  # Обучающие данные
            train_y,  # Обучающие метки
            epochs=epochs,  # Количество эпох
            batch_size=batch_size,  # Размер пакета
            validation_data=(test_x, test_y),  # Данные для валидации
        ),
        model  # Возвращаем модель
    )


# Обучение модели с разными размерами пакетов
history_128, _ = train(batch_size=128, epochs=20)  # Обучение с batch_size=128
history_256, _ = train(batch_size=256, epochs=20)  # Обучение с batch_size=256
history_512, _ = train(batch_size=512, epochs=20)  # Обучение с batch_size=512

# %%
# Построение графиков точности
plt.figure(figsize=(14, 4))

for i, log in enumerate([history_128, history_256, history_512]):
    plt.subplot(1, 3, i + 1)  # Создаем подграфик

    # График точности на обучающих и тестовых данных
    plt.plot(log.history["binary_accuracy"], label="Обучение")
    plt.plot(log.history["val_binary_accuracy"], label="Тест")
    plt.title(f"Динамика точности (batch_size = {128 << i})", pad=10)  # Заголовок графика
    plt.ylabel("Accuracy")  # Ось Y - точность
    plt.xlabel("Эпоха")  # Ось X - эпохи
    plt.legend()  # Легенда
    plt.grid(linestyle="--", alpha=0.6)  # Сетка
    plt.ylim(0.825, 1.0)  # Ограничение по оси Y
    plt.xticks(range(0, 20, 2))  # Разметка оси X

plt.tight_layout()  # Автоматическая настройка расположения графиков
plt.show()  # Отображение графиков

# Построение графиков потерь
plt.figure(figsize=(14, 4))
for i, log in enumerate([history_128, history_256, history_512]):
    plt.subplot(1, 3, i + 1)  # Создаем подграфик

    # График потерь на обучающих и тестовых данных
    plt.plot(log.history["loss"], label="Обучение")
    plt.plot(log.history["val_loss"], label="Тест")
    plt.title(f"Динамика потерь (batch_size = {128 << i})", pad=10)  # Заголовок графика
    plt.ylabel("Loss")  # Ось Y - потери
    plt.xlabel("Эпоха")  # Ось X - эпохи
    plt.legend()  # Легенда
    plt.grid(linestyle="--", alpha=0.6)  # Сетка
    plt.ylim(-0.15, 1.3)  # Ограничение по оси Y
    plt.xticks(range(0, 20, 2))  # Разметка оси X

plt.tight_layout()  # Автоматическая настройка расположения графиков
plt.show()  # Отображение графиков

# %%
# Обучение оптимальной модели
log, model = train(batch_size=512, epochs=2)  # Обучение с оптимальным batch_size=512
model.save("imdb.keras")  # Сохранение модели в файл

# Загрузка модели
model = keras.models.load_model("imdb.keras")  # Загрузка модели из файла

# Предсказание на тестовых данных
for i in range(5):
    index = np.random.randint(0, len(test_x))  # Случайный индекс из тестового набора
    prediction = model.predict(test_x[index : index + 1], verbose=0)[0][0]  # Предсказание для случайного отзыва

    # Вывод результатов
    print(f"\nСлучайный отзыв №{i + 1}")
    print(f"- Предсказанное значение: {prediction:.2f}")  # Предсказанное значение (вероятность положительного отзыва)
    print(f"- Истинное значение:      {test_y[i]:.2f}")  # Истинное значение (0 или 1)