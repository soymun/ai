import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from keras.api.datasets import imdb
from keras import models
from keras import layers
import matplotlib.pyplot as plt

# 1. Загрузка набора данных IMDB
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 2. Подготовка данных
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Векторизация обучающих и тестовых данных
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Векторизация меток
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 3. Создание модели
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 4. Компиляция модели
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Создание проверочного набора
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 6. Обучение модели с записью истории
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 7. Построение графиков потерь и точности
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

epochs = range(1, len(loss_values) + 1)

# График потерь
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# График точности
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 8. Определение оптимального числа эпох
# Оптимальное число эпох можно определить по графику, где validation loss перестает уменьшаться
# Например, если validation loss начинает расти после 4 эпох, то оптимальное число эпох = 4

# 9. Обучение новой модели с оптимальным числом эпох
optimal_epochs = 4  # Пример: оптимальное число эпох
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=optimal_epochs, batch_size=512)

# 10. Сохранение модели в файл
model.save('imdb_model.keras')

# 11. Использование модели для предсказаний
predictions = model.predict(x_test)
print(predictions[:10])  # Пример вывода первых 10 предсказаний

# 12. Эксперименты с разным количеством нейронов
# Можно повторить шаги 3-11 с разным количеством нейронов (например, 32, 64) и сравнить результаты

# 13. Эксперименты с разными функциями потерь
# Можно повторить шаги 3-11 с функцией потерь 'mse' вместо 'binary_crossentropy' и сравнить результаты