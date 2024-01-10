import cv2 as cv
import numpy as np


def findContours(img: np.ndarray, original_img: np.ndarray) -> np.ndarray:
    # 找到装甲板灯条轮廓
    contours, hierarchy = cv.findContours(
        img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )
    imgContours = original_img.copy()  # Create a temporary image for drawing contours
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 200:
            # cv.drawContours(imgContours, cnt, -1, (255, 0, 0), 3)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            # print(approx)
            x, y, w, h = cv.boundingRect(approx)
            cv.rectangle(imgContours, (x, y), (x + w, y + h), (204, 0, 0), 3)
            cv.putText(
                imgContours,
                "Stripe",
                (x + 5, y - 10),
                cv.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (204, 0, 0),
                2,
            )
        # TODO: 确定装甲板

    return imgContours
