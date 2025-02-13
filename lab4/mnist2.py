# stage_1_basic_nn.py
import keras
import numpy as np

import os

from sklearn.metrics import accuracy_score

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras import layers
from keras import Sequential
from matplotlib import pyplot as plt
from keras.api.datasets import mnist
from keras.api.utils import to_categorical

np.random.seed()
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Преобразование в вектор
X_train = X_train.reshape((X_train.shape[0], 28 * 28)).astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], 28 * 28)).astype('float32') / 255

#Преобразование в вектор
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Создание нейронной сети
network = Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

#Компиляция модели
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = network.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

pr = network.predict(X_test)

# Convert predictions to class labels
predicted_classes = np.argmax(pr, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(true_classes, predicted_classes)
print(f'Accuracy: {accuracy}')

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()