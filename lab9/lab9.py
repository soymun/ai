import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.api.applications import VGG16
from keras.api.preprocessing import image
from keras.api.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Загрузка предобученной модели VGG16
model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Загрузка и предобработка изображения
img_path = 'img.png'  # Убедитесь, что изображение находится в той же директории
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.0  # Нормализация

# Создание модели для извлечения активаций промежуточных слоёв
layer_outputs = [layer.output for layer in model.layers[:8]]  # Первые 8 слоёв
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Получение активаций
activations = activation_model.predict(img_tensor)

# Визуализация активаций
layer_names = [layer.name for layer in model.layers[:8]]

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]  # Количество фильтров в слое
    size = layer_activation.shape[1]  # Размер карты признаков

    # Создание сетки для визуализации
    n_cols = 8
    n_rows = n_features // n_cols
    display_grid = np.zeros((size * n_rows, size * n_cols))

    for col in range(n_cols):
        for row in range(n_rows):
            channel_image = layer_activation[0, :, :, row * n_cols + col]
            channel_image -= channel_image.mean()  # Нормализация
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[row * size: (row + 1) * size, col * size: (col + 1) * size] = channel_image

    # Отображение карты признаков
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()