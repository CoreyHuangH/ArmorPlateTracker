import cv2
import numpy as np

cap = cv2.VideoCapture("Resources/stream.mp4")
cap.set(3, 640)
cap.set(4, 480)


def nothing(x):
    pass


cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 240)
cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("Hue Max", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("Sat Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Val Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Val Max", "Trackbars", 255, 255, nothing)

cv2.setTrackbarPos("Hue Min", "Trackbars", 73)
cv2.setTrackbarPos("Hue Max", "Trackbars", 118)
cv2.setTrackbarPos("Sat Min", "Trackbars", 86)
cv2.setTrackbarPos("Sat Max", "Trackbars", 226)
cv2.setTrackbarPos("Val Min", "Trackbars", 128)
cv2.setTrackbarPos("Val Max", "Trackbars", 255)


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

    k = cv2.waitKey(1)
    if k == ord("q"):
        cv2.destroyAllWindows()
        break
