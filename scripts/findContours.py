import cv2
import numpy as np


def findContours(img, original_img) -> np.ndarray:
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
