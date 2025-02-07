import cv2
import numpy as np

# Загрузка классификаторов Хаара для лиц и глаз
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Загрузка изображения
image_path = 'img_1.png'  # Укажите путь к вашему изображению
image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого (для работы классификаторов)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Обнаружение лиц на изображении
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Перебор всех обнаруженных лиц и выделение их на изображении
for (x, y, w, h) in faces:
    # Рисуем прямоугольник вокруг лица
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Область изображения, соответствующая лицу
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]

    # Обнаружение глаз в области лица
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # Рисуем прямоугольники вокруг глаз
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# Отображение результата
cv2.imshow('Detected Faces and Eyes', image)
cv2.waitKey(0)  # Ожидание нажатия любой клавиши
cv2.destroyAllWindows()  # Закрытие всех окон