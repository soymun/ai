from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer

# Загрузка данных
imdb = load_files('path_to_imdb_dataset')
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(imdb.data)
y = imdb.target