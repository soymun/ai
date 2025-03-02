import os
import matplotlib.pyplot as plt
from imageai.Detection import ObjectDetection

# Отключение предупреждений OneDNN (если используете TensorFlow)
# OneDNN — это библиотека для оптимизации вычислений, но иногда она может вызывать предупреждения
# Отключаем её, чтобы избежать лишних сообщений в консоли
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Инициализация детектора объектов
# Создаем экземпляр класса ObjectDetection для обнаружения объектов на изображении
detector = ObjectDetection()

# Установка типа модели YOLOv3
# YOLOv3 — это одна из популярных архитектур нейронных сетей для обнаружения объектов
detector.setModelTypeAsYOLOv3()

# Указание пути к файлу модели YOLOv3
# Модель YOLOv3 должна быть предварительно загружена и сохранена в файл "yolov3.pt"
detector.setModelPath("yolov3.pt")

# Загрузка модели
# Загружаем модель YOLOv3 для дальнейшего использования
detector.loadModel()

# Путь к изображению
# Указываем путь к изображению, на котором будем обнаруживать объекты
image_path = 'img_2.png'

# Выполнение распознавания объектов
# Метод detectObjectsFromImage анализирует изображение и обнаруживает объекты
# input_image: путь к исходному изображению
# output_image_path: путь для сохранения изображения с выделенными объектами
detections = detector.detectObjectsFromImage(input_image=image_path, output_image_path="output.jpg")

# Загрузка изображения с выделенными объектами
# Используем библиотеку matplotlib для загрузки обработанного изображения
output_image = plt.imread("output.jpg")

# Отображение изображения с выделенными объектами
# Создаем фигуру для отображения изображения
plt.figure(figsize=(10, 10))  # Устанавливаем размер фигуры 10x10 дюймов
plt.imshow(output_image)  # Отображаем изображение
plt.axis('off')  # Скрываем оси (координатные оси не нужны для визуализации)
plt.show()  # Показываем изображение в окне

# Вывод результатов в консоль
# Перебираем все обнаруженные объекты и выводим информацию о них
for eachObject in detections:
    # eachObject — это словарь, содержащий информацию об объекте:
    # "name" — название объекта (например, "car", "person")
    # "percentage_probability" — вероятность (точность) обнаружения объекта в процентах
    # "box_points" — координаты прямоугольника, ограничивающего объект (x1, y1, x2, y2)
    print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])