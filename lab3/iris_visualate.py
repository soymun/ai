import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
import pandas as pd

# Загрузка данных
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

sns.pairplot(iris_df, hue='target', palette='viridis')
plt.show()