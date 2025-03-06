# %%
"""
## VGG16
"""
import os

# Отключаем оптимизации OneDNN для совместимости
# OneDNN — это библиотека для оптимизации вычислений, но иногда она может вызывать проблемы с совместимостью
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# %%
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.api.applications import VGG16  # Импорт модели VGG16
from keras import layers, models  # Импорт слоев и моделей Keras

# Задаем размеры изображения
HIGHT = 150
WIDTH = 150

# Загрузка модели VGG16 без верхних слоев (include_top=False)
# input_shape=(HIGHT, WIDTH, 3) — входные данные: изображения размером 150x150 с 3 каналами (RGB)
model = VGG16(include_top=False, input_shape=(HIGHT, WIDTH, 3))

"""
Block 1:

Conv2D (64 filters): Первый сверточный слой с 64 фильтрами размером 3x3. Эти фильтры извлекают базовые признаки, такие как границы и текстуры.

Conv2D (64 filters): Второй сверточный слой с 64 фильтрами размером 3x3. Эти фильтры дополнительно обрабатывают признаки, извлеченные первым слоем.

Block 2:

Conv2D (128 filters): Третий сверточный слой с 128 фильтрами размером 3x3. Эти фильтры извлекают более сложные признаки.

Conv2D (128 filters): Четвертый сверточный слой с 128 фильтрами размером 3x3. Эти фильтры дополнительно обрабатывают признаки, извлеченные предыдущим слоем.

Block 3:

Conv2D (256 filters): Пятый сверточный слой с 256 фильтрами размером 3x3. Эти фильтры извлекают еще более сложные и абстрактные признаки.

Conv2D (256 filters): Шестой сверточный слой с 256 фильтрами размером 3x3.

Conv2D (256 filters): Седьмой сверточный слой с 256 фильтрами размером 3x3.

Block 4:

Conv2D (512 filters): Восьмой сверточный слой с 512 фильтрами размером 3x3. Эти фильтры извлекают высокоуровневые признаки.

Conv2D (512 filters): Девятый сверточный слой с 512 фильтрами размером 3x3.

Conv2D (512 filters): Десятый сверточный слой с 512 фильтрами размером 3x3.

Block 5:

Conv2D (512 filters): Одиннадцатый сверточный слой с 512 фильтрами размером 3x3.

Conv2D (512 filters): Двенадцатый сверточный слой с 512 фильтрами размером 3x3.

Conv2D (512 filters): Тринадцатый сверточный слой с 512 фильтрами размером 3x3.
"""

# Загрузка изображения и изменение его размера до 150x150
img = keras.utils.load_img("img.png", target_size=(HIGHT, WIDTH))

# Преобразование изображения в массив numpy и нормализация значений пикселей в диапазон [0, 1]
tensor = keras.utils.img_to_array(img)
tensor = np.expand_dims(tensor, axis=0) / 255.0

# Создание модели для извлечения активаций всех слоев
# inputs=model.input — входные данные модели VGG16
# outputs=[layer.output for layer in model.layers] — выходные данные всех слоев модели
extraction_model = models.Model(
    inputs=model.input, outputs=[layer.output for layer in model.layers]
)

# Проход по всем сверточным слоям (Conv2D) модели VGG16
for name, activation in zip(
    [layer.name for layer in model.layers if isinstance(layer, layers.Conv2D)],  # Имена сверточных слоев
    extraction_model.predict(tensor),  # Активации для каждого слоя
):
    # Получение количества фильтров и размера активаций
    features, size = activation.shape[-1], activation.shape[1]
    max_features = min(features, 64)  # Ограничиваем количество фильтров до 64 для визуализации
    rows = (max_features + 7) // 8  # Количество строк в сетке для визуализации
    cols = 8  # Количество столбцов в сетке для визуализации

    # Создание пустого изображения для визуализации активаций
    pixels = np.zeros((size * rows, size * cols))
    for i in range(max_features):
        row, col = divmod(i, cols)  # Вычисление позиции фильтра в сетке
        img = activation[0, :, :, i]  # Активация для текущего фильтра
        # Нормализация и масштабирование активаций для визуализации
        img = np.clip((img - img.mean()) / img.std() * 64 + 128, 0, 255).astype("uint8")
        # Размещение активации в соответствующей позиции сетки
        pixels[row * size : (row + 1) * size, col * size : (col + 1) * size] = img

    # Масштабирование изображения для отображения
    scale = 1.0 / size
    plt.figure(figsize=(scale * pixels.shape[1], scale * pixels.shape[0]))
    plt.title(name)  # Заголовок графика с именем слоя
    plt.grid(False)  # Отключение сетки
    plt.axis("off")  # Отключение осей
    plt.imshow(pixels, aspect="auto", cmap="viridis")  # Отображение активаций

plt.show()  # Показ всех графиков
