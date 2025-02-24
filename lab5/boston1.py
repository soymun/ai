import pandas as pd

# URL для загрузки данных
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

# Загрузка данных
# Загрузка данных
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

df = pd.read_csv(url, sep='\s+', header=None, names=column_names)

print(df.size)
print(df['MEDV'])