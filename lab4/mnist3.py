# stage_2_cnn.py
import numpy as np

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras import layers
from keras import Sequential
from matplotlib import pyplot as plt
from keras.api.datasets import mnist
from keras.api.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

cnn = Sequential()
#Чтение и преобразование
cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
#Аггрегация данных
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
#Из 2 в одномерного
cnn.add(layers.Flatten())
cnn.add(layers.Dense(64, activation='relu'))
cnn.add(layers.Dense(10, activation='softmax'))

print(cnn.summary())

cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history_cnn = cnn.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

plt.plot(history_cnn.history['loss'], label='Training Loss')
plt.plot(history_cnn.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history_cnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

cnn.save('mnist_cnn_model.keras')