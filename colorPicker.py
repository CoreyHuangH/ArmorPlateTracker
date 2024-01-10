import cv2 as cv
import numpy as np

cap = cv.VideoCapture("Resources/stream.mp4")
cap.set(3, 640)
cap.set(4, 480)

cv.namedWindow("Trackbars")
cv.resizeWindow("Trackbars", 640, 240)
cv.createTrackbar("Hue Min", "Trackbars", 73, 179, lambda x: print(x))
cv.createTrackbar("Hue Max", "Trackbars", 118, 179, lambda x: print(x))
cv.createTrackbar("Sat Min", "Trackbars", 86, 255, lambda x: print(x))
cv.createTrackbar("Sat Max", "Trackbars", 226, 255, lambda x: print(x))
cv.createTrackbar("Val Min", "Trackbars", 128, 255, lambda x: print(x))
cv.createTrackbar("Val Max", "Trackbars", 255, 255, lambda x: print(x))


while True:
    success, img = cap.read()
    if not success:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue
    imgBlur = cv.GaussianBlur(img, (7, 7), 1)
    imgHSV = cv.cvtColor(imgBlur, cv.COLOR_BGR2HSV)

    h_min = cv.getTrackbarPos("Hue Min", "Trackbars")
    h_max = cv.getTrackbarPos("Hue Max", "Trackbars")
    s_min = cv.getTrackbarPos("Sat Min", "Trackbars")
    s_max = cv.getTrackbarPos("Sat Max", "Trackbars")
    v_min = cv.getTrackbarPos("Val Min", "Trackbars")
    v_max = cv.getTrackbarPos("Val Max", "Trackbars")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(imgHSV, lower, upper)

    cv.imshow("Image Blur", imgBlur)
    # cv.imshow("HSV", imgHSV)
    cv.imshow("Mask", mask)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
