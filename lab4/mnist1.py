import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.api.datasets import mnist
from matplotlib import pyplot as plt

np.random.seed()
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('Тренировочный набор данных: ', X_train.shape)
print('Метки тренировочного набора данных :', y_train.shape)
print('Тестовый набор данных :', X_test.shape)
print('Метки тестового набора данных :', y_test.shape)

plt.imshow(X_train[1])

plt.show()
plt.figure()
plt.imshow(X_train[1])
plt.colorbar()
plt.grid(False)
plt.show()
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
plt.show()
