import cv2 as cv
import numpy as np
from preProcessing import preProcessing
from findContours import findContours

cap = cv.VideoCapture("Resources/stream.mp4")
cap.set(3, 640)
cap.set(4, 480)


def main():
    while True:
        success, img = cap.read()
        if not success:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
        # cv.imshow("Original", img)
        imgTresh = preProcessing(img)
        imgContours = findContours(imgTresh, img)
        cv.imshow("Contours", imgContours)
        # cv.imshow("imgThresh", imgTresh)
        if cv.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()
