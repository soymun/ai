from sklearn import datasets
import pandas as pd

# Загрузка данных
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Визуализация первых 5 строк
iris_df.to_csv('iris.csv', index=False)

# Описание структуры данных
print("\nСтруктура данных IRIS:")
print(f"Количество строк: {iris_df.shape[0]}")
print(f"Количество столбцов: {iris_df.shape[1]}")
print(f"Названия признаков: {iris.feature_names}")
print(f"Классы: {iris.target_names}")
print(iris_df['target'])