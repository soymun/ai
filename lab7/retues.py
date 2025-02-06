import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.api.datasets import reuters
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout
from keras.api.utils import to_categorical
from keras.src.legacy.preprocessing.text import Tokenizer
import numpy as np

# Загрузка данных
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

tokenizer = Tokenizer(num_words=10000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

# Преобразование меток в one-hot encoding
num_classes = 46
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10000,)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')