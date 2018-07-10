#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thur July 5 19:50:49 2018

@author: Dongbo Liu
"""


import numpy as np
import os
from PIL import Image
import PIL.ImageOps  as ImageOps



def get_image_paths( filePath):
    path_exp = os.path.expanduser(filePath)
    classes = os.listdir(path_exp)
    classes.sort()
    nrof_classes = len(classes)
    image_path_list = []



    for i in range(nrof_classes):
        class_name = classes[i]
        print("list class_name="+class_name)

        pathName = os.path.join(filePath, class_name)

        #print("pathName=" + pathName)

        images = os.listdir(pathName)
        image_paths = [os.path.join(pathName, img) for img in images]


        for image_path in image_paths:

            image_path_list.append(image_path)
            print("image_path=" + image_path)

    return image_path_list

def get_images_and_labels(image_path_list):

    image_datas = []
    image_labels = []
    nrof_images_total = 0
    removeCount = 0



    #for image_path in image_path_list:
    for image_path in image_path_list:
            filename = os.path.splitext(os.path.split(image_path)[1])[0]

            print("image_path="+image_path)


            print("filename="+filename)


            class_name = image_path[36:37]

            print("class_name=" + class_name)

            try:
                img = Image.open(image_path)
                # 转为灰度图
                img = img.convert("L")

                data = img.getdata()
                data = np.matrix(data, dtype='float') / 255.0
                #new_data = np.reshape(data, (height, width))

                image_label = filename

                # print("image_data=")
                # print(data)
                # print("image_data.shape=")
                # print(data.shape)
                ##  此處用來打標簽
                if class_name == '0':
                    print("0=")
                    image_label = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == '1':
                    print("1=")
                    image_label = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == '2':
                    print("2=")
                    image_label = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == '3':
                    print("3=")
                    image_label = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == '4':
                   print("4=")
                   image_label = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == '5':
                    print("5=")
                    image_label = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == '6':
                    print("6=")
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == '7':
                    print("7=")
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == '8':
                    print("8=")
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == '9':
                    print("9=")
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == 'A':
                    print("A=")
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == 'B':
                    print("B=")
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == 'C':
                    print("C=")
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == 'D':
                    print("D=")
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
                elif class_name == 'E':
                    print("E=")
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
                elif class_name == 'F':
                    print("F=")
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
                elif class_name == 'G':
                    print("G=")
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
                elif class_name == 'H':
                    print("H=")
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
                elif class_name == 'I':
                    print("I=")
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
                elif class_name == 'J':
                    print("J=")
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
                else:
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


                   # image_label.reshape(1,-1)
                 #   print("image_label=")
                 #   print(image_label)
                 #   print("image_label.shape=")
                 #   print(image_label.shape)

                nrof_images_total += 1
                if  nrof_images_total == 1:
                    image_datas = data
                    image_labels = image_label
                else :
                    image_datas = np.concatenate((image_datas, data), axis=0)
                    #image_datas = image_datas + data
                    image_labels = np.concatenate((image_labels, image_label), axis=0)
                    #image_labels = image_labels + image_label
                    #image_datas.append(data)
                    #image_labels.append(image_label)

            except:
                print("exception image_path=")
                print(image_path)
                  #os.remove(image_path)
                print("exception removeCount=")
                print(removeCount)
                removeCount += 1
                continue
            else:
                continue

    return image_datas, image_labels



# older
def get_datas_and_labels( filePath):
    path_exp = os.path.expanduser(filePath)
    classes = os.listdir(path_exp)
    classes.sort()
    nrof_classes = len(classes)
    image_datas = []
    image_labels = []
    nrof_images_total = 0
    removeCount = 0


    for i in range(nrof_classes):
        class_name = classes[i]
       # print("class_name="+class_name)

        pathName = os.path.join(filePath, class_name)

        #print("pathName=" + pathName)

        images = os.listdir(pathName)
        image_paths = [os.path.join(pathName, img) for img in images]

        for image_path in image_paths:


           # print("image_path="+image_path)

            filename = os.path.splitext(os.path.split(image_path)[1])[0]
         #   print("filename="+filename)

            try:
                img = Image.open(image_path)
                # 转为灰度图
                img = img.convert("L")
                img = ImageOps.invert(img)
                data = img.getdata()
                data = np.matrix(data, dtype='float') / 255.0
                #new_data = np.reshape(data, (height, width))

                image_label = filename

               # print("image_data=")
               # print(data)
               # print("image_data.shape=")
               # print(data.shape)

                if class_name == 'A':
                    image_label = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == 'B':
                    image_label = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == 'C':
                    image_label = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
                elif class_name == 'D':
                    image_label = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
                elif class_name == 'E':
                    image_label = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
                elif class_name == 'F':
                    image_label = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
                elif class_name == 'G':
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
                elif class_name == 'H':
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
                elif class_name == 'I':
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
                elif class_name == 'J':
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
                else:
                    image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

               # image_label.reshape(1,-1)
             #   print("image_label=")
             #   print(image_label)
             #   print("image_label.shape=")
             #   print(image_label.shape)

                nrof_images_total += 1
                if  nrof_images_total == 1:
                    image_datas = data
                    image_labels = image_label
                else :
                    image_datas = np.concatenate((image_datas, data), axis=0)
                #image_datas = image_datas + data
                    image_labels = np.concatenate((image_labels, image_label), axis=0)
                #image_labels = image_labels + image_label
                #image_datas.append(data)
                #image_labels.append(image_label)

            except:
                print("exception image_path=")
                print(image_path)
                #os.remove(image_path)
                print("exception removeCount=")
                print(removeCount)
                removeCount += 1
                continue
            else:
                continue

    return image_datas, image_labels, nrof_images_total ,removeCount



