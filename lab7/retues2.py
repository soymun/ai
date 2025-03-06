import os

# Отключаем оптимизации OneDNN для TensorFlow для совместимости
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Импорт необходимых модулей из Keras и других библиотек
from keras.api.datasets import reuters  # Набор данных Reuters

# Загрузка данных Reuters
# num_words=10000 — ограничиваем словарь 10,000 наиболее часто встречающихся слов
(x_train, y_train), (x_test, y_test) = reuters.load_data()

total_news = len(x_train) + len(x_test)
print(f"Общее количество новостей: {total_news}")

# Пример для первой новости в обучающем наборе
word_index = reuters.get_word_index()  # Получаем словарь слов и их индексов
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])  # Инвертируем словарь

# Преобразуем последовательность индексов в текст
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])

# Считаем количество символов
num_characters = len(decoded_newswire)
print(f"Количество символов в первой новости: {num_characters}")