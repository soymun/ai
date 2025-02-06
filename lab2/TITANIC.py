import pandas as pd

# Загрузка данных
url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'
titanic_df = pd.read_csv(url)

# Визуализация первых 5 строк
print(titanic_df.head())

# Описание структуры данных
print("\nСтруктура данных Titanic:")
print(f"Количество строк: {titanic_df.shape[0]}")
print(f"Количество столбцов: {titanic_df.shape[1]}")
print(f"Названия столбцов: {titanic_df.columns.tolist()}")