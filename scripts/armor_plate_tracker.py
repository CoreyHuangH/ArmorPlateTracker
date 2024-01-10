import cv2
import numpy as np

cap = cv2.VideoCapture(
    "/Users/huangyichuan/workspace/Project/stripeDetect-py/Resources/stream.mp4"
)
cap.set(3, 640)
cap.set(4, 480)


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
    # cv2.imshow("mask", mask)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    imgGray = cv2.bitwise_and(imgGray, imgGray, mask=mask)  # Mask the grayscale image
    imgThresh = cv2.adaptiveThreshold(
        imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )  # Adaptive thresholding
    return imgThresh


def findContours(img, original_img):
    # 找到装甲板灯条轮廓
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    imgContours = original_img.copy()  # Create a temporary image for drawing contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            # cv2.drawContours(imgContours, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(approx)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContours, (x, y), (x + w, y + h), (204, 0, 0), 3)
            cv2.putText(
                imgContours,
                "Stripe",
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (204, 0, 0),
                2,
            )
    return imgContours


def calculateArmorPosePNP():
    # TODO: 利用PNP解算解算出装甲板在相机坐标系下的空间位置和朝向角
    pass


def kalmanPredict():
    # TODO: 建立匀速运动模型的卡尔曼滤波预测出装甲板在1s后的位置
    pass


while True:
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    # cv2.imshow("Original", img)
    imgTresh = preProcessing(img)
    imgContours = findContours(imgTresh, img)
    cv2.imshow("Contours", imgContours)
    # cv2.imshow("imgThresh", imgTresh)
    if cv2.waitKey(1) == ord("q"):
        break
