import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# Загрузка данных
reuters = fetch_20newsgroups(subset='all')
reuters_df = pd.DataFrame({'text': reuters.data, 'target': reuters.target})

# Визуализация первых 5 статей
print(reuters_df.head())

# Описание структуры данных
print("\nСтруктура данных Reuters:")
print(f"Количество статей: {reuters_df.shape[0]}")
print(f"Количество категорий: {len(reuters.target_names)}")