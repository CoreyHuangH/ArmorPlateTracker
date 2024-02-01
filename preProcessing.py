import cv2
import numpy as np


def preProcessing(img):
    # 图片预处理
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)  # Gaussian blur
    imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)  # Convert to HSV color space

    h_min, h_max = 73, 118
    s_min, s_max = 86, 226
    v_min, v_max = 128, 255

    lower = np.array([h_min, s_min, v_min])  # Lower HSV threshold
    upper = np.array([h_max, s_max, v_max])  # Upper HSV threshold

    mask = cv2.inRange(imgHSV, lower, upper)  # Create mask
    _, mask = cv2.threshold(
        mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )  # Threshold mask

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    imgGray = cv2.bitwise_and(imgGray, imgGray, mask=mask)  # Mask the grayscale image

    kernel = np.ones((5, 5), dtype=np.uint8)
    imgDilation = cv2.dilate(imgGray, kernel, iterations=3)  # Dilation
    imgEroded = cv2.erode(imgDilation, kernel, iterations=1)  # Erosion

    _, imgTresh = cv2.threshold(imgEroded, 0, 255, cv2.THRESH_BINARY)

    # cv2.imshow("imgThresh", imgTresh)

    return imgTresh
