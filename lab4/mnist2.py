# stage_1_basic_nn.py
import os

import numpy as np
from sklearn.metrics import accuracy_score

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras import layers
from keras import Sequential
from matplotlib import pyplot as plt
from keras.api.datasets import mnist
from keras.api.utils import to_categorical

np.random.seed()
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование данных
# Изменяем форму данных из 2D (28x28) в 1D (784) и нормализуем значения пикселей в диапазоне [0, 1]
X_train = X_train.reshape((X_train.shape[0], 28 * 28)).astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], 28 * 28)).astype('float32') / 255

# Преобразование меток в one-hot encoding
# Например, метка 5 становится [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Создание модели нейронной сети
network = Sequential()  # Инициализация последовательной модели

# Добавление полносвязного слоя с 512 нейронами и функцией активации ReLU
# input_shape=(28 * 28,) указывает, что на вход подается вектор из 784 элементов (28x28)
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

# Добавление выходного слоя с 10 нейронами (по одному для каждого класса) и функцией активации softmax
# softmax преобразует выходные значения в вероятности принадлежности к каждому классу
network.add(layers.Dense(10, activation='softmax'))

# Компиляция модели
# optimizer='rmsprop' - алгоритм оптимизации для обновления весов
# loss='categorical_crossentropy' - функция потерь для многоклассовой классификации
# metrics=['accuracy'] - метрика, которая будет вычисляться во время обучения (точность)
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
# X_train, y_train - обучающие данные и метки
# epochs=5 - количество эпох обучения (полных проходов по данным)
# batch_size=128 - количество образцов, обрабатываемых за один шаг обновления весов
# validation_split=0.2 - доля данных, используемых для валидации (20% обучающих данных)
history = network.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# Предсказание на тестовых данных
# network.predict(X_test) возвращает вероятности для каждого класса
pr = network.predict(X_test)

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
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')  # Ось X - количество эпох
plt.ylabel('Loss')  # Ось Y - значение функции потерь
plt.legend()  # Добавление легенды
plt.show()  # Отображение графика

# График точности (accuracy) на обучающих и валидационных данных
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')  # Ось X - количество эпох
plt.ylabel('Accuracy')  # Ось Y - значение точности
plt.legend()  # Добавление легенды
plt.show()  # Отображение графика

first_layer_weights = network.layers[0].get_weights()[0]

# Визуализация весов
fig, axes = plt.subplots(4, 4, figsize=(10, 10))  # Создаем сетку 4x4 для отображения весов
fig.suptitle('Weights of the First Layer', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < 512:  # Убедимся, что мы не выходим за пределы количества нейронов
        # Визуализируем веса для i-го нейрона
        ax.imshow(first_layer_weights[:, i].reshape(28, 28), cmap='viridis')
        ax.axis('off')  # Отключаем оси

plt.tight_layout()
plt.show()


