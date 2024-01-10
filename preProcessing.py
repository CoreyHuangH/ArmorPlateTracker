import cv2 as cv
import numpy as np


def preProcessing(img: np.ndarray) -> np.ndarray:
    # 图片预处理
    imgBlur = cv.GaussianBlur(img, (7, 7), 1)  # Gaussian blur
    imgHSV = cv.cvtColor(imgBlur, cv.COLOR_BGR2HSV)  # Convert to HSV color space
    h_min, h_max = 73, 118
    s_min, s_max = 86, 226
    v_min, v_max = 128, 255
    lower = np.array([h_min, s_min, v_min])  # Lower HSV threshold
    upper = np.array([h_max, s_max, v_max])  # Upper HSV threshold
    mask = cv.inRange(imgHSV, lower, upper)  # Create mask
    # cv.imshow("mask", mask)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to grayscale
    imgGray = cv.bitwise_and(imgGray, imgGray, mask=mask)  # Mask the grayscale image
    imgThresh = cv.adaptiveThreshold(
        imgGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2
    )  # Adaptive thresholding
    return imgThresh
