import cv2
import numpy as np
import os


IMG_PATH = "D:/liudongbo/dataset/Test/20180719/1.jpg"
SAVE_PATH = "D:/liudongbo/dataset/Test/20180719/save/"
WIDTH_START = 2
WIDTH_END = 5
HIGHT_START = 2
HIGHT_END = 5


def split(img_path,imageName):

    img = cv2.imread(img_path,0)
    ret,thresh = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
    #列求和
    col= thresh.sum(axis = 0)
    #行求和
    lin = thresh.sum(axis = 1)

    #存储坐标
    loc = np.zeros((2,2),dtype = int)

    #是否取到坐标标志
    flag1 =flag2 =flag3=flag4=0
    for i in range(len(col)):
        if col[i] != 0 & flag1 ==0:
           loc[0][0] = i
           flag1 = 1
           print(col[i])
        if col[len(col)-i-1] != 0 & flag2 == 0:
           loc[0][1] =  len(col)-i-1
           flag2 = 1
           print(col[len(col)-i-1])

    for j in range(len(lin)):
        if lin[j] !=0 & flag3==0:
            loc[1][0] = j
            flag3= 1
            print(lin[j])
        if lin[len(lin)-j-1] !=0 & flag4==0:
            loc[1][1] = len(lin) - j -1
            flag4 = 1
            print(len(lin)-j-1)

    crop_img= thresh[loc[1][1]-HIGHT_START:loc[1][0]+HIGHT_END,loc[0][1]-WIDTH_START:loc[0][0]+WIDTH_END]  #thresh[86:1548, 0:323]
    ret,image1 = cv2.threshold(crop_img,200,255,cv2.THRESH_BINARY_INV)
    #保存图片
    if not os.path.isdir(SAVE_PATH):
       os.makedirs(SAVE_PATH)
    cv2.imwrite(SAVE_PATH + imageName + ".jpg", image1)

    cv2.imshow("draw_img", image1)   #thresh
    cv2.waitKey(0)

def main():
    split(IMG_PATH,"123")

if __name__ == '__main__':
    main()








