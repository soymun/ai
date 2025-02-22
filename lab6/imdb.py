import os

# Отключаем оптимизации OneDNN для TensorFlow, чтобы избежать потенциальных проблем с совместимостью
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from keras.api.datasets import imdb  # Импорт набора данных IMDB
from keras import models  # Импорт модуля для создания моделей
from keras import layers  # Импорт модуля для создания слоев нейронной сети
import matplotlib.pyplot as plt  # Импорт библиотеки для построения графиков

# 1. Загрузка набора данных IMDB
# Загружаем данные IMDB, ограничивая словарь 10000 наиболее часто встречающихся слов
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# 2. Подготовка данных
def vectorize_sequences(sequences, dimension=10000):
    """
    Векторизация последовательностей (преобразование текста в бинарный вектор)

    :param sequences: Список последовательностей (индексов слов)
    :param dimension: Размерность вектора (по умолчанию 10000)
    :return: Массив бинарных векторов
    """
    results = np.zeros(
        (len(sequences), dimension))  # Создаем нулевой массив размером (количество последовательностей, dimension)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # Устанавливаем 1 на позициях, соответствующих индексам слов в последовательности
    return results


# Векторизация обучающих и тестовых данных
x_train = vectorize_sequences(train_data)  # Преобразуем обучающие данные в бинарные векторы
x_test = vectorize_sequences(test_data)  # Преобразуем тестовые данные в бинарные векторы

# Векторизация меток
y_train = np.asarray(train_labels).astype('float32')  # Преобразуем метки обучающих данных в массив float32
y_test = np.asarray(test_labels).astype('float32')  # Преобразуем метки тестовых данных в массив float32

# 3. Создание модели
model = models.Sequential()  # Инициализация последовательной модели

# Добавление полносвязного слоя с 32 нейронами и функцией активации ReLU
# input_shape=(10000,) указывает, что на вход подается вектор из 10000 элементов
model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))

# Добавление еще одного полносвязного слоя с 32 нейронами и функцией активации ReLU
model.add(layers.Dense(32, activation='relu'))

# Добавление выходного слоя с 1 нейроном и функцией активации sigmoid
# sigmoid преобразует выходное значение в вероятность (для бинарной классификации)
model.add(layers.Dense(1, activation='sigmoid'))

# 4. Компиляция модели
# optimizer='rmsprop' - алгоритм оптимизации для обновления весов
# loss='binary_crossentropy' - функция потерь для бинарной классификации
# metrics=['accuracy'] - метрика, которая будет вычисляться во время обучения (точность)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Создание проверочного набора
# Выделяем первые 10000 образцов из обучающих данных для валидации
x_val = x_train[:10000]
partial_x_train = x_train[10000:]  # Остальные данные используются для обучения
y_val = y_train[:10000]  # Метки для валидации
partial_y_train = y_train[10000:]  # Метки для обучения

# 6. Обучение модели с записью истории
# Обучение модели на части обучающих данных с использованием валидационных данных
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=5,  # Количество эпох обучения
                    batch_size=512,  # Размер пакета (количество образцов за один шаг обновления весов)
                    validation_data=(x_val, y_val))  # Данные для валидации

# 7. Построение графиков потерь и точности
history_dict = history.history  # Получаем историю обучения
loss_values = history_dict['loss']  # Значения потерь на обучающих данных
val_loss_values = history_dict['val_loss']  # Значения потерь на валидационных данных
acc_values = history_dict['accuracy']  # Значения точности на обучающих данных
val_acc_values = history_dict['val_accuracy']  # Значения точности на валидационных данных

epochs = range(1, len(loss_values) + 1)  # Создаем диапазон эпох для построения графиков

# График потерь
plt.plot(epochs, loss_values, 'bo', label='Training loss')  # Потери на обучающих данных
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')  # Потери на валидационных данных
plt.title('Training and validation loss')  # Заголовок графика
plt.xlabel('Epochs')  # Ось X - количество эпох
plt.ylabel('Loss')  # Ось Y - значение функции потерь
plt.legend()  # Добавление легенды
plt.show()  # Отображение графика

# График точности
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')  # Точность на обучающих данных
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')  # Точность на валидационных данных
plt.title('Training and validation accuracy')  # Заголовок графика
plt.xlabel('Epochs')  # Ось X - количество эпох
plt.ylabel('Accuracy')  # Ось Y - значение точности
plt.legend()  # Добавление легенды
plt.show()  # Отображение графика

# 10. Сохранение модели в файл
model.save('imdb_model.keras')  # Сохраняем модель в файл для последующего использования

# 11. Использование модели для предсказаний
predictions = model.predict(x_test)  # Получаем предсказания для тестовых данных
print(predictions[:10])  # Пример вывода первых 10 предсказаний (вероятности положительного отзыва)