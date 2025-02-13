import numpy as np

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.api.models import load_model
from keras.api.datasets import mnist
from keras.api.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование данных
# Изменяем форму данных из 2D (28x28) в 3D (28x28x1) для использования в сверточных слоях
# Нормализуем значения пикселей в диапазоне [0, 1]
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255

# Преобразование меток в one-hot encoding
# Например, метка 5 становится [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
y_test = to_categorical(y_test)

# Загрузка модели
# Загружаем предварительно обученную модель из файла 'mnist_cnn_model.keras'
model = load_model('mnist_cnn_model.keras')

# Предсказание на тестовых данных
# model.predict(X_test) возвращает вероятности для каждого класса
# predictions - это массив, где каждая строка содержит вероятности для каждого из 10 классов
predictions = model.predict(X_test)

# Преобразование вероятностей в метки классов
# np.argmax возвращает индекс максимального значения (класс с наибольшей вероятностью)
# predicted_labels - массив предсказанных меток
predicted_labels = np.argmax(predictions, axis=1)

# Вывод предсказанных и фактических меток для первых 10 примеров
for i in range(10):
    # np.argmax(y_test[i]) преобразует one-hot encoding обратно в метку класса
    print(f"Предсказанная цифра: {predicted_labels[i]}, Фактическая цифра: {np.argmax(y_test[i])}")