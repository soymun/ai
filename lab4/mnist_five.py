import numpy as np

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.api.models import load_model
from keras.api.datasets import mnist
from keras.api.utils import to_categorical

# Загрузка данных MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Предобработка данных
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255
y_test = to_categorical(y_test)

# Загрузка сохраненной модели
model = load_model('mnist_cnn_model.keras')

# Выполнение предсказаний
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Пример вывода предсказаний
for i in range(10):
    print(f"Предсказанная цифра: {predicted_labels[i]}, Фактическая цифра: {np.argmax(y_test[i])}")