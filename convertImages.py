# -*- coding:utf-8 -*-

import numpy as np
import cv2
from PIL import Image
import os

SOURCE_DATA_DIR = "D:/liudongbo/dataset/train_set_02/"
DESTINATION_DATA_DIR = "D:/liudongbo/dataset/train_set_2val"

def  convertImage(imgPath):

      im1 = cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
      print("im2=")
      print(im1)
      im2 = cv2.GaussianBlur(im1, (3, 3), 0)
      _,im3 = cv2.threshold(im2,150,255,cv2.THRESH_BINARY )
      im4 = cv2.resize(im3,(28,28))
      print("im3=")
      print(im4)
      #cv2.imwrite("D:/liudongbo/dataset/9.jpg",im4)
      return im4

def  convertAllImages(sourceFilePath,destinationFilePath):

     path_exp = os.path.expanduser(sourceFilePath)
     classes = os.listdir(path_exp)
     classes.sort()
     nrof_classes = len(classes)
     image_path_list = []

     for i in range(nrof_classes):
         class_name = classes[i]
         print("list class_name=" + class_name)

         sourcePathName = os.path.join(sourceFilePath, class_name)

        # print("pathName=" + pathName)

         images = os.listdir(sourcePathName)
         image_paths = [os.path.join(sourcePathName, img) for img in images]
         for image_path in image_paths:
             img = convertImage(image_path)
             print("img=")
             print(img)

            # time.sleep(30)

             destinationPathName = os.path.join(destinationFilePath, class_name)

             print("before  destinationPathName=")
             print(destinationPathName)

            # time.sleep(30)

             filename = os.path.splitext(os.path.split(image_path)[1])[0]
             print("filename = ")
             print(filename)

             isExists = os.path.exists(destinationPathName)
             if not isExists:
                 os.makedirs(destinationPathName)

             destinationPathName = destinationPathName + '/' + filename + '.jpg'

             print("after  destinationPathName=")
             print(destinationPathName)

            # time.sleep(30)
             cv2.imwrite(destinationPathName, img)



def main(argv=None):
    convertAllImages(SOURCE_DATA_DIR,DESTINATION_DATA_DIR)



if __name__ == '__main__':
    main()

