import cv2
import numpy as np

def nothing(x):
    pass

img = cv2.imread(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\pillow\images\256x256_tiger.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Original", img)
cv2.namedWindow("Result")

cv2.createTrackbar("Gaussian_k", "Result", 1, 31, nothing)  
cv2.createTrackbar("BlockSize", "Result", 3, 51, nothing)  
cv2.createTrackbar("C", "Result", 0, 20, nothing)          
cv2.createTrackbar("Th1", "Result", 50, 300, nothing)       
cv2.createTrackbar("Th2", "Result", 150, 300, nothing)     

while True:

    k = cv2.getTrackbarPos("Gaussian_k", "Result")

    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1

    blur = cv2.GaussianBlur(gray, (k, k), 0)


    block = cv2.getTrackbarPos("BlockSize", "Result")
    c = cv2.getTrackbarPos("C", "Result")

    if block < 3:
        block = 3
    if block % 2 == 0:
        block += 1

    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block,
        c
    )


    t1 = cv2.getTrackbarPos("Th1", "Result")
    t2 = cv2.getTrackbarPos("Th2", "Result")

    edges = cv2.Canny(binary, t1, t2)


    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)


    cv2.imshow("Result", result)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()