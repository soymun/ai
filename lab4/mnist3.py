# stage_2_cnn.py
import numpy as np

import os

from sklearn.metrics import accuracy_score

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras import layers
from keras import Sequential
from matplotlib import pyplot as plt
from keras.api.datasets import mnist
from keras.api.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование данных
# Изменяем форму данных из 2D (28x28) в 3D (28x28x1) для использования в сверточных слоях
# Нормализуем значения пикселей в диапазоне [0, 1]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255

# Преобразование меток в one-hot encoding
# Например, метка 5 становится [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Создание модели сверточной нейронной сети (CNN)
cnn = Sequential()  # Инициализация последовательной модели

# Добавление сверточного слоя (Conv2D)
# 32 - количество фильтров (выходных каналов)
# (3, 3) - размер ядра свертки (3x3)
# activation='relu' - функция активации ReLU (Rectified Linear Unit)
# input_shape=(28, 28, 1) - форма входных данных (28x28 пикселей, 1 канал)
cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Добавление слоя подвыборки (MaxPooling2D)
# (2, 2) - размер окна для агрегации (уменьшает размерность данных в 2 раза)
cnn.add(layers.MaxPooling2D((2, 2)))

# Добавление второго сверточного слоя
# 64 - количество фильтров
# (3, 3) - размер ядра свертки
# activation='relu' - функция активации ReLU
cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Добавление второго слоя подвыборки
cnn.add(layers.MaxPooling2D((2, 2)))

# Добавление третьего сверточного слоя
cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Преобразование 3D-данных в 1D (выравнивание)
# Flatten слой преобразует данные в одномерный вектор для передачи в полносвязные слои
cnn.add(layers.Flatten())

# Добавление полносвязного слоя (Dense)
# 64 - количество нейронов
# activation='relu' - функция активации ReLU
cnn.add(layers.Dense(64, activation='relu'))

# Добавление выходного слоя
# 10 - количество нейронов (по одному для каждого класса)
# activation='softmax' - функция активации softmax (преобразует выходные значения в вероятности)
cnn.add(layers.Dense(10, activation='softmax'))

# Вывод структуры модели
print(cnn.summary())

# Компиляция модели
# optimizer='rmsprop' - алгоритм оптимизации RMSprop (адаптивный метод градиентного спуска)
# loss='categorical_crossentropy' - функция потерь для многоклассовой классификации
# metrics=['accuracy'] - метрика, которая будет вычисляться во время обучения (точность)
cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
# X_train, y_train - обучающие данные и метки
# epochs=5 - количество эпох обучения (полных проходов по данным)
# batch_size=128 - количество образцов, обрабатываемых за один шаг обновления весов
# validation_split=0.2 - доля данных, используемых для валидации (20% обучающих данных)
history_cnn = cnn.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# Предсказание на тестовых данных
# network.predict(X_test) возвращает вероятности для каждого класса
pr = cnn.predict(X_test)

# Преобразование вероятностей в метки классов
# np.argmax возвращает индекс максимального значения (класс с наибольшей вероятностью)
predicted_classes = np.argmax(pr, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Вычисление точности
# accuracy_score сравнивает предсказанные метки с истинными и возвращает долю правильных ответов
accuracy = accuracy_score(true_classes, predicted_classes)
print(f'Accuracy: {accuracy}')


# Построение графиков
# График потерь (loss) на обучающих и валидационных данных
plt.plot(history_cnn.history['loss'], label='Training Loss')
plt.plot(history_cnn.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')  # Ось X - количество эпох
plt.ylabel('Loss')  # Ось Y - значение функции потерь
plt.legend()  # Добавление легенды
plt.show()  # Отображение графика

# График точности (accuracy) на обучающих и валидационных данных
plt.plot(history_cnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')  # Ось X - количество эпох
plt.ylabel('Accuracy')  # Ось Y - значение точности
plt.legend()  # Добавление легенды
plt.show()  # Отображение графика

# Сохранение модели в файл
cnn.save('mnist_cnn_model.keras')