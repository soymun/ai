# %%
"""
# IMDB
"""
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# %%
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.api import layers, optimizers, losses, metrics, activations

# Подготовка данных
# Перекодирование в векторы 0-1, бинарную матрицу
def vectorize(sequences):
    results = np.zeros((len(sequences), NUM_WORDS))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


# Параметры для обучения модели
NUM_WORDS = 10000
INITIAL_LR = 0.001

# Загрузка датасета IMDB
(data, labels), (data2, labels2) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

# Объединение датасетов
data = np.concatenate((data, data2), axis=0)
labels = np.concatenate((labels, labels2), axis=0)

# Перемешивание данных
indices = np.arange(data.shape[0])
np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]

# Векторизация меток
x = vectorize(data)
y = np.asarray(labels).astype("float32")

# Выбор первых 10000 отзывов для тестового набора данных
# Выбор остальных 40000 отзывов для обучающего набора данных
train_x, train_y, test_x, test_y = x[10000:], y[10000:], x[:10000], y[:10000]


def train(batch_size, epochs):
    # Добавление слоев для модели
    model = keras.Sequential(
        [
            layers.Input(shape=(NUM_WORDS,)),
            layers.Dense(32, activation=activations.relu),
            layers.Dropout(0.3),
            layers.Dense(16, activation=activations.relu),
            layers.Dropout(0.25),
            layers.Dense(1, activation=activations.sigmoid),
        ]
    )

    # Компиляция модели
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=INITIAL_LR),
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy()],
    )

    # Обучение модели с использованием данных IMDB
    return (
        model.fit(
            train_x,
            train_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(test_x, test_y),
        ),
        model
    )


history_128, _ = train(batch_size=128, epochs=20)
history_256, _ = train(batch_size=256, epochs=20)
history_512, _ = train(batch_size=512, epochs=20)

# %%
plt.figure(figsize=(14, 4))

for i, log in enumerate([history_128, history_256, history_512]):
    plt.subplot(1, 3, i + 1)

    plt.plot(log.history["binary_accuracy"])
    plt.plot(log.history["val_binary_accuracy"])
    plt.title(f"Динамика точности (batch_size = {128 << i})", pad=10)
    plt.ylabel("Accuracy")
    plt.xlabel("Эпоха")
    plt.legend(["Обучение", "Тест"])
    plt.grid(linestyle="--", alpha=0.6)
    plt.ylim(0.825, 1.0)
    plt.xticks(range(0, 20, 2))
    

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 4))
for i, log in enumerate([history_128, history_256, history_512]):
    plt.subplot(1, 3, i + 1)

    plt.plot(log.history["loss"])
    plt.plot(log.history["val_loss"])
    plt.title(f"Динамика потерь (batch_size = {128 << i})", pad=10)
    plt.ylabel("Loss")
    plt.xlabel("Эпоха")
    plt.legend(["Обучение", "Тест"])
    plt.grid(linestyle="--", alpha=0.6)
    plt.ylim(-0.15, 1.3)
    plt.xticks(range(0, 20, 2))

plt.tight_layout()
plt.show()

# %%
# Обучение оптимальной модели
log, model = train(batch_size=512, epochs=2)
model.save("imdb.keras")

# Загрузка модели
model = keras.models.load_model("imdb.keras")

# Предсказание
for i in range(5):
    index = np.random.randint(0, len(test_x))
    prediction = model.predict(test_x[index : index + 1], verbose=0)[0][0]

    print(f"\nСлучайный отзыв №{i + 1}")
    print(f"- Предсказанное значение: {prediction:.2f}")
    print(f"- Истинное значение:      {test_y[i]:.2f}")