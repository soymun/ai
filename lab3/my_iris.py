import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Загрузка датасета
my_df = pd.read_csv("data_set.csv")

#Добавление обозначений
data = my_df[['X', 'Y']]

#Разделение данных
X_train, X_test, y_train, y_test = train_test_split(data, my_df['Target'], random_state=0)

#Создание класификатора
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#Сохранение в файл
filename = 'my_model.sav'
with open(filename, 'wb') as file:
    pickle.dump(knn, file)

#Предсказания
pr = knn.predict(X_test)

print("Прогноз вида на тестовом наборе:\n {}".format(pr))
print("Точность прогноза на тестовом наборе:{:.2f}".format(np.mean(pr==y_test)))

with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

#Предсказания по нашим данным
X_new1 = np.array([[0.75, 1], [3, -2], [6, -4]])
X_new_df = pd.DataFrame(X_new1, columns=['X', 'Y'])
result = loaded_model.predict(X_new_df)

print("По сохранённой модели:")
print(result)