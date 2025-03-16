import cv2
import numpy as np


def show_res(frame_name, res):
    cv2.imshow(frame_name, res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# загрузка изображения
image = cv2.imread("D:/cv_projects/lab2/fruits_vegs.jpg")

show_res("Original Image", image)

# задаем количество кластеров
k = int(input("Введите количество кластеров (k): "))

# приведение формы изображения к двумерному массиву пикселей
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)  # преобразование во float

# определение критериев завершения и выполнение алгоритма k-средних
criteria_end = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(
    pixel_values, k, None, criteria_end, 10, cv2.KMEANS_RANDOM_CENTERS
)

# преобразование центров обратно в uint8
centers = np.uint8(centers)

# создание итогового изображения, где каждый пиксель имеет цвет соответствующего кластера
final_image = centers[labels.flatten()]  # делаем из labels одномерный массив
final_image = final_image.reshape(image.shape)

# отображение итогового изображения
show_res("Final Image", final_image)
