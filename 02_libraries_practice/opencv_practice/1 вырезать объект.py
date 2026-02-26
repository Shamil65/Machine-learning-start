import cv2


img = cv2.imread(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\ежик.jpg", cv2.IMREAD_GRAYSCALE)

template = img[350:500, 50:250]

cv2.imshow("большое фото", img)
cv2.imshow("маленькое фото", template)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\template.png", template)
cv2.imwrite(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\img.png", img)

