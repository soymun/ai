from sklearn import datasets
import pandas as pd

boston = datasets.load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['PRICE'] = boston.target

# Визуализация первых 5 строк
boston_df.to_csv('boston.csv', index=False)

# Описание структуры данных
print("\nСтруктура данных Boston:")
print(f"Количество строк: {boston_df.shape[0]}")
print(f"Количество столбцов: {boston_df.shape[1]}")
print(f"Названия признаков: {boston.feature_names}")