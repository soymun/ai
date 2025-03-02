import os
# Отключаем оптимизации OneDNN для совместимости
# OneDNN — это библиотека для оптимизации вычислений, но иногда она может вызывать проблемы с совместимостью
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Импорт библиотек
from keras import models  # Импорт модуля для создания моделей
from keras.api.applications import VGG16  # Импорт предварительно обученной модели VGG16
from keras.api.preprocessing import image  # Импорт модуля для работы с изображениями
import keras.api.utils  # Импорт утилит Keras
import numpy as np  # Импорт библиотеки для работы с массивами
import matplotlib.pyplot as plt  # Импорт библиотеки для визуализации

# Загрузка предварительно обученной модели VGG16
# weights='imagenet' — используем веса, обученные на наборе данных ImageNet
# include_top=False — исключаем полносвязные слои на вершине сети (используем только сверточные слои)
# input_shape=(150, 150, 3) — задаем размер входного изображения (150x150 пикселей, 3 канала — RGB)
model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Загрузка изображения
img_path = 'img.png'  # Указываем путь к изображению
# Загружаем изображение и изменяем его размер до 150x150 пикселей
img = keras.utils.load_img(img_path, target_size=(150, 150))
# Преобразуем изображение в массив numpy
img_tensor = keras.utils.img_to_array(img)
# Добавляем дополнительную ось (batch dimension), чтобы изображение соответствовало формату входных данных модели
img_tensor = np.expand_dims(img_tensor, axis=0)
# Нормализация изображения (приведение значений пикселей к диапазону [0, 1])
img_tensor /= 255.0
# Выводим форму тензора изображения (1, 150, 150, 3) — 1 изображение, 150x150 пикселей, 3 канала (RGB)
print("Форма тензора изображения:", img_tensor.shape)

# Создание модели для извлечения промежуточных активаций
# Берем выходы первых 8 слоев модели VGG16
layer_outputs = [layer.output for layer in model.layers[:8]]
# Создаем новую модель, которая принимает входные данные VGG16 и возвращает выходы первых 8 слоев
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# Получение активаций для входного изображения
# Передаем изображение в модель и получаем активации (выходы) для каждого из первых 8 слоев
activations = activation_model.predict(img_tensor)

# Визуализация карт признаков
# Получаем имена первых 8 слоев
layer_names = [layer.name for layer in model.layers[:8]]
# Количество карт признаков в строке для визуализации
images_per_row = 16

# Перебираем слои и их активации
for layer_name, layer_activation in zip(layer_names, activations):
    # Количество карт признаков в текущем слое
    n_features = layer_activation.shape[-1]
    # Размер карты признаков (ширина и высота)
    size = layer_activation.shape[1]

    # Количество столбцов для визуализации
    n_cols = n_features // images_per_row
    # Создаем сетку для отображения карт признаков
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # Заполняем сетку картами признаков
    for col in range(n_cols):
        for row in range(images_per_row):
            # Извлекаем карту признаков
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            # Нормализация карты признаков
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            # Ограничиваем значения пикселей диапазоном [0, 255] и преобразуем в целые числа
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            # Размещаем карту признаков в сетке
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    # Масштабируем изображение для отображения
    scale = 1. / size
    # Создаем фигуру для визуализации
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    # Заголовок — имя слоя
    plt.title(layer_name)
    # Скрываем сетку
    plt.grid(False)
    # Отображаем карты признаков
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

# Показываем все изображения
plt.show()