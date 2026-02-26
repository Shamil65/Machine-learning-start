import cv2
import numpy as np
import time  

def ncc(img, templ):
    h, w = templ.shape
    rows, cols = img.shape[0] - h + 1, img.shape[1] - w + 1
    res = np.zeros((rows, cols))
    mean_templ = np.mean(templ)
    
    for y in range(rows):
        for x in range(cols):
            region = img[y:y + h, x:x + w]
            mean_reg = np.mean(region)

            num = np.sum((region - mean_reg) * (templ - mean_templ))
            den = np.sqrt(np.sum((region - mean_reg) ** 2) * np.sum((templ - mean_templ) ** 2))

            res[y, x] = num / den if den != 0 else 0

    return res

def main():
    img = cv2.imread(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\ежик.jpg", cv2.IMREAD_GRAYSCALE)
    templ = cv2.imread(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\template.png", cv2.IMREAD_GRAYSCALE)

    start_time = time.time()  
    res = ncc(img, templ)
    end_time = time.time()    

    print(f"Время выполнения NCC: {end_time - start_time:.4f} секунд")

    max_correlation_index = np.argmax(res)
    top_left = np.unravel_index(max_correlation_index, res.shape)
    max_correlation_value = res.item(max_correlation_index)

    print("Максимальная корреляция в точке:", top_left)
    print("Значение корреляции:", max_correlation_value)

    img_color = cv2.imread(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\ежик.jpg")
    h, w = templ.shape
    bottom_right = (top_left[1] + w, top_left[0] + h)
    cv2.rectangle(img_color, (top_left[1], top_left[0]), bottom_right, (0, 0, 255), 2)

    vis_res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\корреляция.png", vis_res)
    cv2.imwrite(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\совпадение.png", img_color)

    cv2.imshow("Корреляция", vis_res)
    cv2.imshow("Совпадение", img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()