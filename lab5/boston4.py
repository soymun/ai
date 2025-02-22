import pandas as pd

# URL для загрузки данных
# Ссылка на набор данных о ценах на жилье в Бостоне, размещенный в репозитории UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

# Загрузка данных
# Список названий столбцов для DataFrame
column_names = [
    'CRIM',  # Уровень преступности на душу населения
    'ZN',    # Доля жилых участков, отведенных под дома с большими участками
    'INDUS', # Доля не розничных торговых площадей в городе
    'CHAS',  # Наличие реки (1 - есть, 0 - нет)
    'NOX',   # Концентрация оксидов азота
    'RM',    # Среднее количество комнат в доме
    'AGE',   # Доля домов, построенных до 1940 года
    'DIS',   # Взвешенное расстояние до пяти рабочих центров Бостона
    'RAD',   # Индекс доступности радиальных магистралей
    'TAX',   # Ставка налога на имущество
    'PTRATIO', # Соотношение учеников и учителей в школах
    'B',     # Доля афроамериканского населения
    'LSTAT', # Доля населения с низким статусом
    'MEDV'   # Медианная стоимость дома (целевая переменная)
]

# Загрузка данных в DataFrame
# sep='\s+' - разделитель (пробелы или табуляции)
# header=None - отсутствие заголовка в файле
# names=column_names - задаем имена столбцов
df = pd.read_csv(url, sep='\s+', header=None, names=column_names)

# Вычисление матрицы корреляции
# corr() вычисляет попарные корреляции между всеми столбцами DataFrame
# Результат - матрица, где каждая ячейка показывает корреляцию между двумя признаками
corr_matrix = df.corr()

# Выбор корреляции целевой переменной (MEDV) с другими признаками
# corr_matrix['MEDV'] возвращает корреляции между MEDV и всеми остальными признаками
# sort_values(ascending=False) сортирует значения по убыванию
corr_with_target = corr_matrix['MEDV'].sort_values(ascending=False)

# Вывод корреляций
print(corr_with_target)

# Выбор наиболее значимых признаков
# corr_with_target.index[1:5] выбирает индексы (названия признаков) с 1 по 4 (исключая MEDV)
# Это признаки с наибольшей корреляцией с целевой переменной
top_features = corr_with_target.index[1:5]

# Вывод наиболее значимых признаков
print("Наиболее значимые признаки:", top_features)