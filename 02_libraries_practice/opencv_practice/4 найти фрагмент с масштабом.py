import cv2
import numpy as np

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
    scale_factor = 1.05

    img = cv2.imread(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\ежик.jpg", cv2.IMREAD_GRAYSCALE)
    templ = cv2.imread(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\template.png", cv2.IMREAD_GRAYSCALE)

    img_scaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    templ_scaled = cv2.resize(templ, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    res = ncc(img_scaled, templ_scaled)

    max_correlation_index = np.argmax(res)
    top_left = np.unravel_index(max_correlation_index, res.shape)
    max_correlation_value = res.item(max_correlation_index)

    print("Максимальная корреляция в точке:", top_left)
    print("Значение корреляции:", max_correlation_value)

    img_color = cv2.imread(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\ежик.jpg")

    img_color_scaled = cv2.resize(img_color, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)


    h, w = templ_scaled.shape
    bottom_right = (top_left[1] + w, top_left[0] + h)
    cv2.rectangle(img_color_scaled, (top_left[1], top_left[0]), bottom_right, (0, 0, 255), 2)

    vis_res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\3 корреляция.png", vis_res)
    cv2.imwrite(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\3 совпадение.png", img_color_scaled)


    cv2.imshow("Корреляция", vis_res)
    cv2.imshow("Совпадение", img_color_scaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()