import cv2
import numpy as np

cap = cv2.VideoCapture("Resources/stream.mp4")
cap.set(3, 640)
cap.set(4, 480)

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 240)
cv2.createTrackbar("Hue Min", "Trackbars", 73, 179, lambda x: print(x))
cv2.createTrackbar("Hue Max", "Trackbars", 118, 179, lambda x: print(x))
cv2.createTrackbar("Sat Min", "Trackbars", 86, 255, lambda x: print(x))
cv2.createTrackbar("Sat Max", "Trackbars", 226, 255, lambda x: print(x))
cv2.createTrackbar("Val Min", "Trackbars", 128, 255, lambda x: print(x))
cv2.createTrackbar("Val Max", "Trackbars", 255, 255, lambda x: print(x))


while True:
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
    h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
    s_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
    s_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
    v_min = cv2.getTrackbarPos("Val Min", "Trackbars")
    v_max = cv2.getTrackbarPos("Val Max", "Trackbars")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)

    cv2.imshow("Image Blur", imgBlur)
    # cv2.imshow("HSV", imgHSV)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
