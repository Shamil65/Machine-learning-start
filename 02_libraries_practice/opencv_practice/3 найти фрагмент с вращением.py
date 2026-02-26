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

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return rotated

def main():
    angle = 4

    img = cv2.imread(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\ежик.jpg", cv2.IMREAD_GRAYSCALE)
    templ = cv2.imread(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\template.png", cv2.IMREAD_GRAYSCALE)

    img_color = cv2.imread(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\ежик.jpg")

    img_rotated = rotate_image(img, angle)
    templ_rotated = rotate_image(templ, angle)
    img_color_rotated = rotate_image(img_color, angle)

    res = ncc(img_rotated, templ_rotated)

    max_correlation_index = np.argmax(res)
    top_left = np.unravel_index(max_correlation_index, res.shape)
    max_correlation_value = res.item(max_correlation_index)

    print("Максимальная корреляция в точке:", top_left)
    print("Значение корреляции:", max_correlation_value)

    h, w = templ_rotated.shape
    bottom_right = (top_left[1] + w, top_left[0] + h)
    cv2.rectangle(img_color_rotated, (top_left[1], top_left[0]), bottom_right, (0, 0, 255), 2)

    vis_res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\2 корреляция.png", vis_res)
    cv2.imwrite(r"C:\Users\ZiganshinShamil\Desktop\Study\ml_study\02_libraries_practice\opencv_practice\images\2 совпадение.png", img_color_rotated)

    cv2.imshow("Корреляция", vis_res)
    cv2.imshow("Совпадение", img_color_rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()