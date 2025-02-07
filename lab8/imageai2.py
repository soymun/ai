import os
import matplotlib.pyplot as plt
from imageai.Detection import ObjectDetection

# Отключение предупреждений OneDNN (если используете TensorFlow)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Инициализация детектора
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolov3.pt")  # Укажите путь к модели YOLOv3
detector.loadModel()

# Путь к изображению
image_path = 'img_1.png'  # Замените на путь к вашему изображению

# Выполнение распознавания объектов
detections = detector.detectObjectsFromImage(input_image=image_path, output_image_path="output.jpg")

# Загрузка изображения с выделенными объектами
output_image = plt.imread("output.jpg")

# Отображение изображения с выделенными объектами
plt.figure(figsize=(10, 10))
plt.imshow(output_image)
plt.axis('off')  # Скрыть оси
plt.show()

# Вывод результатов в консоль
for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])