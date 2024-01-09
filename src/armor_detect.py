import cv2
import numpy as np
from math import sqrt


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
                # print(approx)
                # print(len(approx))
                # objCor = len(approx)
                print("Approx 0 shape:", approx[0].squeeze().shape)
                print("Approx 1 shape:", approx[1].squeeze().shape)

                x, y, w, h = cv2.boundingRect(approx)

                if len(approx) >= 2:
                    left_contour = approx[:4].reshape(-1, 2)
                    right_contour = approx[4:].reshape(-1, 2)
                    determineArmor(left_contour, right_contour)

                if w / h < 0.8:
                    cv2.rectangle(imgContours, (x, y), (x + w, y + h), (204, 0, 0), 3)
    cv2.imshow("Contours", imgContours)


def determineArmor(left_contour, right_contour):
    if len(left_contour) >= 2 and len(right_contour) >= 2:
        hull0 = cv2.convexHull(np.int0(left_contour))
        hull1 = cv2.convexHull(np.int0(right_contour))

        rect0 = cv2.minAreaRect(hull0)
        rect1 = cv2.minAreaRect(hull1)

        h0, w0 = rect0[1]
        h1, w1 = rect1[1]

        angel0 = np.arctan2(
            left_contour[0][1] - left_contour[1][1],
            left_contour[0][0] - left_contour[1][0],
        )
        angel1 = np.arctan2(
            right_contour[0][1] - right_contour[1][1],
            right_contour[0][0] - right_contour[1][0],
        )
        ur0, vr0 = left_contour[0][0], left_contour[0][1]
        ur1, vr1 = left_contour[1][0], left_contour[1][1]
        ul0, vl0 = right_contour[0][0], right_contour[0][1]
        ul1, vl1 = right_contour[1][0], right_contour[1][1]

        x1 = min(h0, h1) / max(h0, h1)
        x2 = min(w0, w1) / max(w0, w1)
    else:
        x1 = 0
        x2 = 0

    x3 = angel0 - angel1
    x4 = max(h0, h1) / sqrt(
        ((ur0 + ur1 - ul0 - ul1) / 2) ** 2 + ((vr0 + vr1 - vl0 - vl1) / 2) ** 2
    )

    if len(left_contour) >= 2 and len(right_contour) >= 2:
        x5 = (
            (ul1 - ul0) * ((ur0 + ur1 - ul0 - ul1) / 2)
            + (vl1 - vl0) * ((vr0 + vr1 - vl0 - vl1) / 2)
        ) / sqrt(
            ((ul1 - ul0) ** 2 + (vl1 - vl0) ** 2) * ((ur0 + ur1 - ul0 - ul1) / 2) ** 2
            + ((vr0 + vr1 - vl0 - vl1) / 2) ** 2
        )
    else:
        x5 = 0

    if x1 > 0.5 and x2 > 0.5 and abs(x3) < 0.1 and x4 > 0.8 and abs(x5) < 0.1:
        print("Armor detected!")
    else:
        print("Not armor.")


while True:
    # _, img = cap.read()
    imgStatic = cv2.imread(
        "/Users/huangyichuan/workspace/cvProject/armorDetection/Resources/armor.jpg"
    )
    imgThresh = preProcessing(imgStatic)
    findContours(imgThresh, imgStatic)
    # cv2.imshow("imgThresh", imgThresh)
    if cv2.waitKey(1) == ord("q"):
        break
