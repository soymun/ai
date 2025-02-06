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

# Визуализация корреляционной матрицы
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.show()