import cv2
import numpy as np

from preProcessing import preProcessing
from findContours import findContours

cap = cv2.VideoCapture(
    "/Users/huangyichuan/workspace/Project/stripeDetect-py/Resources/stream.mp4"
)
cap.set(3, 640)
cap.set(4, 480)


def main():
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


if __name__ == "__main__":
    main()
