import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# URL для загрузки данных
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

# Загрузка данных
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
df = pd.read_csv(url, sep='\s+', header=None, names=column_names)

corr_matrix = df.corr()

corr_with_target = corr_matrix['MEDV'].sort_values(ascending=False)
print(corr_with_target)

# Наиболее значимые признаки
top_features = corr_with_target.index[1:5]  # Исключаем MEDV
print("Наиболее значимые признаки:", top_features)