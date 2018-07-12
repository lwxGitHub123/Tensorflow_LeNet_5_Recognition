# -*- coding:utf-8 -*-
import cv2
import os




SOURCE_DATA_DIR = "D:/liudongbo/dataset/train_set_02/"
DESTINATION_DATA_DIR = "D:/liudongbo/dataset/train_set_2val/"


def transmit2value(imgPath):
    img1 = cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE )
    _,img2 = cv2.threshold(img1,50,255,cv2.THRESH_BINARY )
    return img2


def readDataAndLabel(sourceFilePath,destinationFilePath):
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
            img = transmit2value(image_path)
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
            #img.save(destinationPathName)

def main(argv=None):
    readDataAndLabel(SOURCE_DATA_DIR,DESTINATION_DATA_DIR)



if __name__ == '__main__':
    main()

# img1,img2 = transmit2value("3_2.png")
# img = np.hstack((img1,img2))
# cv2.namedWindow("img", 0)
# cv2.resizeWindow("img", 1400, 700)
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()