import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():

    img_name = '../img/star.png'

    imageGRAY = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    imageBGR = cv2.imread(img_name, cv2.IMREAD_COLOR)

    blur = cv2.GaussianBlur(imageGRAY, (1, 1), 0)
    ret1, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    frame = cv2.drawContours(imageBGR, contours, -1, (0,0,0), 10)

    # 出力
    cv2.imwrite('../img/blur.png', blur)
    cv2.imwrite('../img/output.png', frame)

    count = []
    for i in contours :
        x = 0
        y = 0
        for n in i :
            x += n[0][0]
            y += n[0][1]
        count.append([x/len(i), y/len(i)])
    count = np.array(count)

    print("結果 : " + str(len(count)) + "個")

if __name__ == '__main__':
    main()
