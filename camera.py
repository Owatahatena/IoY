# -*- coding: utf-8 -*-
import cv2
import numpy as np
from time import sleep


def find_rects(mask):
    rects = []

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    for contour in contours:
        approx = cv2.convexHull(contour)
        rect = cv2.boundingRect(approx)
        rects.append(np.array(rect))
    return rects[:1]



def sabun(gray,bg):
    th = 30    # 差分画像の閾値
    mask = cv2.absdiff(gray, bg)
    # 差分画像を二値化してマスク画像を算出
    mask[mask < th] = 0
    mask[mask >= th] = 255
    mask = cv2.GaussianBlur(mask, ksize=(3,3), sigmaX=1.3)
    return mask

def judge(image, rects):
    img = image
    for rect in rects:
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        rect_img = img[y:y+height, x:x+width]
        #cv2.imshow("rect", rect_img)

        hsv = cv2.cvtColor(rect_img, cv2.COLOR_BGR2HSV_FULL)

        lower_bake = np.array([0, 0, 0])
        upper_bake = np.array([190, 190, 180])
        img_mask = cv2.inRange(hsv, lower_bake, upper_bake)
        img_mask = cv2.GaussianBlur(img_mask, ksize=(3,3), sigmaX=1.5)

        img_color = cv2.bitwise_and(rect_img, rect_img, mask=img_mask)
        #cv2.imshow("ans",img_color)

        imgray = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
        #cv2.imshow("aaaa",imgray)
        lower_bake = np.array([0, 22, 100])
        upper_bake = np.array([190, 200, 180])
        thresh = cv2.inRange(hsv, lower_bake, upper_bake)
        thresh = cv2.GaussianBlur(thresh, ksize=(3,3), sigmaX=1.5)
        # ret,thresh = cv2.threshold(imgray,50,170,0)
        # thresh = cv2.GaussianBlur(thresh, ksize=(3,3), sigmaX=1.5)
        #cv2.imshow("thresh",thresh)
        #cv2.imshow("aaaa",thresh)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(hierarchy)
        contours.sort(key=cv2.contourArea, reverse=True)
        #print(len(contours))

        out_img = cv2.drawContours(rect_img, contours, 0, (0,255,0), 3)
        #cv2.imshow("out",out_img)
        ans = abs(x - width) * abs(y - height)
        #print(ans)
        if (contours):
            area = cv2.contourArea(contours[0])


            if(area >= ans / 2):
                #cv2.bitwise_and(img,img,mask = thresh)
                #cv2.copyTo(img, thresh)
                img = cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (255, 0, 0), thickness=2)

        else:
            pass

        cv2.imshow("aaa",img)
        k = cv2.waitKey(1)

    return img



def main():
    i = 0      # カウント変数
    j = 0
    th = 30    # 差分画像の閾値

    # カメラのキャプチャ
    cap = cv2.VideoCapture(0)

    # 最初のフレームを背景画像に設定
    ret, bg = cap.read()
    # グレースケール変換
    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)

    while(True):
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mask = sabun(gray,bg)
        rects = find_rects(mask)
        img = judge(frame, rects)

        sleep(0.050)


    capture.release()
    cv2.destroyAllWindows()

        # qキーが押されたら途中終了


if __name__ == '__main__':
    main()
