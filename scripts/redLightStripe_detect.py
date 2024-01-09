import cv2
import numpy as np


# cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 150)


def preProcessing(img):
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1.8)  # Gaussian blur
    imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    h_min, h_max = 19, 45
    s_min, s_max = 0, 67
    v_min, v_max = 239, 255
    lower = np.array([h_min, s_min, v_min])  # Lower HSV threshold
    upper = np.array([h_max, s_max, v_max])  # Upper HSV threshold
    mask = cv2.inRange(imgHSV, lower, upper)  # Binary thresholding
    # cv2.imshow("mask", mask)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    imgGray = cv2.bitwise_and(imgGray, imgGray, mask=mask)  # Mask the grayscale image
    # 在灰度图上做阈值分割，分割出目标区域的二值图像
    imgThresh = cv2.adaptiveThreshold(
        imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return imgThresh


def findContours(img, original_img):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    imgContours = original_img.copy()  # 创建临时图像用于绘制轮廓
    for cnt in contours:
        if cnt is not None:
            area = cv2.contourArea(cnt)
            print(area)
            if area > 350:
                # cv2.drawContours(imgContours, cnt, -1, (255, 0, 0), 3)
                peri = cv2.arcLength(cnt, True)
                # print(peri)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                print(approx)
                # print(len(approx))
                # objCor = len(approx)
                x, y, w, h = cv2.boundingRect(approx)
                if w / h < 0.8:
                    cv2.rectangle(imgContours, (x, y), (x + w, y + h), (204, 0, 0), 3)
    cv2.imshow("Contours", imgContours)


while True:
    # _, img = cap.read()
    imgStatic = cv2.imread("Resources/armor.jpg")
    imgThresh = preProcessing(imgStatic)
    findContours(imgThresh, imgStatic)
    # cv2.imshow("imgThresh", imgThresh)
    if cv2.waitKey(1) == ord("q"):
        break
