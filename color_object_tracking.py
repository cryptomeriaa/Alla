import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def nothing(x):
    pass


cv2.namedWindow("Trackbars")
# настроены параметры цвета апельсина
cv2.createTrackbar("L - H", "Trackbars", 12, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 153, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 64, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 25, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 231, 255, nothing)

old_center = (0, 0) # переменная, которая примет координаты центра при смене кадров

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # присвоение параметров маске
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower_orange = np.array([l_h, l_s, l_v])
    upper_orange = np.array(([u_h, u_s, u_v]))
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # избавление от шума для лучшей обводки


    # нахождение контуров и координат центра
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2] # выделение внешнего контура
    center = (0, 0)

    # если контур обнаружен
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea) #поиск контура максимального размера (апельсин > мандарин)
        ((x, y), radius) = cv2.minEnclosingCircle(c) #минимальный контур, обводящий объект полностью
        M = cv2.moments(c) 
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2) #отображение контура
            cv2.circle(frame, center, 5, (255, 0, 0), -1) # отображение центра
            cv2.putText(frame, "centroid", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1) # настройка текста: расположение, шрифт, цвет и пр.

            # определение направления движения (влево/вправо)
            if old_center[0] < center[0]:
                cv2.putText(frame, "Object is moving to the left", (10, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            elif old_center[0] > center[0]:
                cv2.putText(frame, "Object is moving to the right", (10, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        old_center = center

    cv2.imshow("Frame", frame)
    #cv2.imshow("Mask", mask)  Раскомментировать при настройке параметров цвета



    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()