import cv2
import numpy as np
import time

def main():
    img_path = r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\ежик.jpg"
    templ_path = r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\template.png"

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    templ = cv2.imread(templ_path, cv2.IMREAD_GRAYSCALE)


    start_time = time.time()
    res = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)
    end_time = time.time()

    print(f"Время выполнения cv2.matchTemplate: {end_time - start_time:.4f} секунд")

 
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    max_correlation_value = max_val

    print("Максимальная корреляция в точке:", top_left)
    print("Значение корреляции:", max_correlation_value)

    img_color = cv2.imread(img_path)
    h, w = templ.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img_color, top_left, bottom_right, (0, 0, 255), 2)

  
    vis_res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\корреляция_cv2.png", vis_res)
    cv2.imwrite(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\совпадение_cv2.png", img_color)

    cv2.imshow("Корреляция cv2", vis_res)
    cv2.imshow("Совпадение cv2", img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()