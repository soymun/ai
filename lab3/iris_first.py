import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
#import species as species
from pandas.plotting import scatter_matrix
from sklearn.datasets import  load_iris
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
# *****
iris_dataset=load_iris()
#print(iris_dataset)
print('\n\n 1 Часть. Характеристики датафрейма IRIS:\n')
print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))
print("Тип массива data: {}".format(type(iris_dataset['data'])))
print("Форма массива data: {}".format(iris_dataset['data'].shape))
print("Цель: {}".format(iris_dataset['target']))
print("Название ответов: {}".format(iris_dataset['target_names']))
print(iris_dataset['DESCR'][:193]+"\n...")
print("Название признаков: \n{}".format(iris_dataset['feature_names']))
print("Расположение файла: \n{}".format(iris_dataset['filename']))
print("Пять первых строк массива data:\n{}".format(iris_dataset['data'][:5]))
print("Пять первых строк массива data с 2 столбцами:\n{}".format(iris_dataset['data'][:5,2:]))
print("Правильные ответы:\n{}".format(iris_dataset['target']))
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],
iris_dataset['target'],random_state=0)
# ****
print("Размерность массива X_train: {}".format(X_train.shape))
print("Размерность массива y_train: {}".format(y_train.shape))
print("Размерность массива X_test: {}".format(X_test.shape))
print("Размерность массива y_test: {}".format(y_test.shape))
#  *****
print("\n 2 Часть. Обучение классификатора и вывод результатов:\n")
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
#Практическое использование классификатора
X_new=np.array([[5,2.9,1,0.2]])
pr=knn.predict(X_new)
# save the model to disk
filename = 'finalized_obuch_model.sav'
pickle.dump(knn, open(filename, 'wb'))
print ("Метка вида цветка: {}".format(pr))
print("Вид цветка: {}".format(iris_dataset['target_names'][pr]))
pr=knn.predict(X_test)
print("Прогноз вида на тестовом наборе:\n {}".format(pr))
print("Точность прогноза на тестовом наборе:{:.2f}".format(np.mean(pr==y_test)))
print(' Обученная модель:')

#print([knn.fit(X_train,y_train)])
loaded_model = pickle.load(open(filename, 'rb'))
X_new1=np.array([[5.9, 3. , 5.1, 1.8],[6.9, 3.1, 4.9, 1.5]])
result = loaded_model.predict(X_new1)
print("По сохранённой модели:")
print(result)
print("Вид цветка: {}".format(iris_dataset['target_names'][result]))