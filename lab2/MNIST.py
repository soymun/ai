import pandas as pd
from sklearn.datasets import fetch_openml

# Загрузка данных
mnist = fetch_openml('mnist_784', version=1)
mnist_df = pd.DataFrame(mnist.data)
mnist_df['target'] = mnist.target

# Визуализация первых 5 строк
print(mnist_df.head())

# Описание структуры данных
print("\nСтруктура данных MNIST:")
print(f"Количество изображений: {mnist_df.shape[0]}")
print(f"Количество признаков (пикселей): {mnist_df.shape[1] - 1}")
print(f"Классы: {mnist.target.unique()}")