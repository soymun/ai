from sklearn import datasets
import pandas as pd

# Загрузка данных
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Перемешивание данных
shuffled_iris_df = iris_df.sample(frac=1, random_state=42)

# Визуализация первых 5 строк перемешанных данных
print("Первые 5 строк перемешанных данных:")
print(shuffled_iris_df.head())

# Описание структуры данных
print("\nСтруктура данных IRIS:")
print(f"Количество строк: {shuffled_iris_df.shape[0]}")
print(f"Количество столбцов: {shuffled_iris_df.shape[1]}")
#Названия признаков: ['длина чашелистика (см)', 'ширина чашелистика (см)', 'длина лепестка (см)', 'ширина лепестка (см)']
print(f"Названия признаков: {iris.feature_names}")
#Классы: ['сетоса', 'версиколор', 'виргиника']
print(f"Классы: {iris.target_names}")
print(shuffled_iris_df)