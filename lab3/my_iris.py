import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Sample data creation for demonstration purposes
my_df = pd.read_csv("data_set.csv")

# Selecting columns X and Y
data = my_df[['X', 'Y']]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, my_df['Target'], random_state=0)

# Initializing and training the KNN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Saving the model to a file
filename = 'my_model.sav'
with open(filename, 'wb') as file:
    pickle.dump(knn, file)

# Making predictions with the trained model
pr = knn.predict(X_test)

# Loading the model from the file
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Making predictions with new data points
X_new1 = np.array([[2, 3], [5, 3]])
X_new_df = pd.DataFrame(X_new1, columns=['X', 'Y'])
result = loaded_model.predict(X_new_df)

print("По сохранённой модели:")
print(result)