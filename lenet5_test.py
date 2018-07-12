import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import lenet5_infernece
import lenet5_train
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import PIL.ImageOps  as ImageOps


img_num = [0] * 20


def evaluate(X_test, y_test_lable, My_Yd):
    with tf.Graph().as_default() as g:
        # 定義輸出為4維矩陣的placeholder
        x_ = tf.placeholder(tf.float32, [None, lenet5_train.INPUT_NODE], name='x-input')
        x = tf.reshape(x_, shape=[-1, 28, 28, 1])

        y_ = tf.placeholder(tf.float32, [None, lenet5_train.OUTPUT_NODE], name='y-input')

        regularizer = tf.contrib.layers.l2_regularizer(lenet5_train.REGULARIZATION_RATE)
        y = lenet5_infernece.inference(x, False, regularizer,False)  #tf.AUTO_REUSE
        global_step = tf.Variable(0, trainable=False)

        # Evaluate model
        pred_max = tf.argmax(y, 1)
        y_max = tf.argmax(y_, 1)
        correct_pred = tf.equal(pred_max, y_max)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        batchsize = 20
        test_batch_len = int(X_test.shape[0] / batchsize)
        test_acc = []

        test_xs = np.reshape(X_test, (
            X_test.shape[0],
            lenet5_train.IMAGE_SIZE,
            lenet5_train.IMAGE_SIZE,
            lenet5_train.NUM_CHANNELS))

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        # saver = tf.train.import_meta_graph("./mnist/mnist_model.meta")
        with tf.Session() as sess:
            saver.restore(sess, "D:/liudongbo/models/Mnist_models/lenet5_model")

            My_test_pred = sess.run(pred_max, feed_dict={x: test_xs[:20]})
            print("期望值：", My_Yd)
            print("before cnvert My_test_pred =")
            print(My_test_pred)
            My_test_pred1 = []
            for i in range(len(My_test_pred)):
                print("My_test_pred[i] =")
                print(My_test_pred[i])


                if My_test_pred[i] == 0:
                    My_test_pred1.append(0)
                elif My_test_pred[i] == 1:
                    My_test_pred1.append(1)
                    print("A=")
                elif My_test_pred[i] == 2:
                    My_test_pred1.append(2)
                elif My_test_pred[i] == 3:
                    My_test_pred1.append(3)
                elif My_test_pred[i] == 4:
                    My_test_pred1.append(4)
                elif My_test_pred[i] == 5:
                    My_test_pred1.append(5)
                elif My_test_pred[i] == 6:
                    My_test_pred1.append(6)
                elif My_test_pred[i] == 7:
                    My_test_pred1.append(7)
                elif My_test_pred[i] == 8:
                    My_test_pred1.append(8)
                elif My_test_pred[i] == 9:
                    My_test_pred1.append(9)
                elif My_test_pred[i] == 10:
                    My_test_pred1.append('A')
                elif My_test_pred[i] == 11:
                    My_test_pred1.append('B')
                elif My_test_pred[i] == 12:
                    My_test_pred1.append('C')
                elif My_test_pred[i] == 13:
                    My_test_pred1.append('D')
                elif My_test_pred[i] == 14:
                    My_test_pred1.append('E')
                elif My_test_pred[i] == 15:
                    My_test_pred1.append('F')
                elif My_test_pred[i] == 16:
                    My_test_pred1.append('G')
                elif My_test_pred[i] == 17:
                    My_test_pred1.append('H')
                elif My_test_pred[i] == 18:
                    My_test_pred1.append('I')
                elif My_test_pred[i] == 19:
                    My_test_pred1.append('J')
                else:
                    My_test_pred1.append('--')

            print("預測值：", My_test_pred1)
            My_acc = sess.run(accuracy, feed_dict={x: test_xs, y_: y_test_lable})
            print('Test accuracy: %.2f%%' % (My_acc * 100))
            display_result(My_test_pred,My_test_pred1)
            return


def display_result(my_prediction,my_prediction_alh):
    img_res = [0] * 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(20):
        img_res[i] = np.zeros((64, 64, 3), np.uint8)
        img_res[i][:, :] = [255, 255, 255]
        if (my_prediction[i] % 20) == (i % 20):
            cv2.putText(img_res[i], str(my_prediction_alh[i]), (15, 52), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            cv2.putText(img_res[i], str(my_prediction_alh[i]), (15, 52), font, 2, (255, 0, 0), 3, cv2.LINE_AA)

    Input_Numer_name = ['Input 0', 'Input 1', 'Input 2', 'Input 3', 'Input 4', \
                        'Input 5', 'Input 6', 'Input 7', 'Input8', 'Input9', \
                        'Input 0', 'Input 1', 'Input 2', 'Input 3', 'Input 4', \
                        'Input 5', 'Input 6', 'Input 7', 'Input8', 'Input9',
                        ]

    predict_Numer_name = ['predict 0', 'predict 1', 'predict 2', 'predict 3', 'predict 4', \
                          'predict 5', 'predict6 ', 'predict 7', 'predict 8', 'predict 9', \
                          'predict 0', 'predict 1', 'predict 2', 'predict 3', 'predict 4', \
                          'predict 5', 'predict6 ', 'predict 7', 'predict 8', 'predict 9',
                          ]

    for i in range(20):
        if i < 10:
            plt.subplot(4, 10, i + 1), plt.imshow(img_num[i], cmap='gray')
            plt.title(Input_Numer_name[i]), plt.xticks([]), plt.yticks([])
            plt.subplot(4, 10, i + 11), plt.imshow(img_res[i], cmap='gray')
            plt.title(predict_Numer_name[i]), plt.xticks([]), plt.yticks([])
        else:
            plt.subplot(4, 10, i + 11), plt.imshow(img_num[i], cmap='gray')
            plt.title(Input_Numer_name[i]), plt.xticks([]), plt.yticks([])
            plt.subplot(4, 10, i + 21), plt.imshow(img_res[i], cmap='gray')
            plt.title(predict_Numer_name[i]), plt.xticks([]), plt.yticks([])

    plt.show()


def main(argv=None):
    #### Loading the data
    # 自己手寫的20個數字
    My_X = np.zeros((20, 784), dtype=int)
    # 自己手寫的20個數字對應的期望數字
    My_Yd = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=int)
    #My_Yd_OneHot = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=int)
    #My_Yd_OneHot = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
    #My_Yd =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    # 輸入20個手寫數字圖檔28x28=784 pixel，
    Input_Numer = [0] * 20
    Input_Numer[0] = "D:/liudongbo/dataset/recognition/0/mnist_test_3.jpg"
    Input_Numer[1] = "D:/liudongbo/dataset/recognition/1/mnist_test_5.jpg"
    Input_Numer[2] = "D:/liudongbo/dataset/recognition/2/mnist_test_1.jpg"
    Input_Numer[3] = "D:/liudongbo/dataset/recognition/3/mnist_test_30.jpg"
    Input_Numer[4] = "D:/liudongbo/dataset/recognition/4/mnist_test_19.jpg"
    Input_Numer[5] = "D:/liudongbo/dataset/recognition/5/mnist_test_15.jpg"
    Input_Numer[6] = "D:/liudongbo/dataset/recognition/6/mnist_test_21.jpg"
    Input_Numer[7] = "D:/liudongbo/dataset/recognition/7/mnist_test_17.jpg"
    Input_Numer[8] = "D:/liudongbo/dataset/recognition/8/mnist_test_84.jpg"
    Input_Numer[9] = "D:/liudongbo/dataset/recognition/9/mnist_test_9.jpg"
    Input_Numer[10] = "D:/liudongbo/dataset/recognition/9/zhangzhiqiang1.jpg"
    Input_Numer[11] = "D:/liudongbo/dataset/recognition/8/mnist_test_1415.jpg"
    Input_Numer[12] = "D:/liudongbo/dataset/recognition/7/mnist_test_702.jpg"
    Input_Numer[13] = "D:/liudongbo/dataset/recognition/6/mnist_test_568.jpg"
    Input_Numer[14] = "D:/liudongbo/dataset/recognition/5/mnist_test_509.jpg"
    Input_Numer[15] = "D:/liudongbo/dataset/recognition/4/mnist_test_475.jpg"  #"D:/liudongbo/dataset/recognition/alh/MTIgQ29uY29yZGUgSXRhbGljIDEzMzE5LnR0Zg==.png"
    Input_Numer[16] = "D:/liudongbo/dataset/recognition/3/mnist_test_548.jpg" #"D:/liudongbo/dataset/recognition/alh/dGVlbiBzcGlyaXQudHRm.png"
    Input_Numer[17] = "D:/liudongbo/dataset/recognition/2/mnist_test_477.jpg" #"D:/liudongbo/dataset/recognition/alh/Q2FuYWRpYW5QaG90b2dyYXBoZXIub3Rm.png"
    Input_Numer[18] = "D:/liudongbo/dataset/recognition/1/mnist_test_168.jpg" #"D:/liudongbo/dataset/recognition/alh/TGltZXJpY2stRGVtaUJvbGQub3Rm.png"
    Input_Numer[19] = "D:/liudongbo/dataset/recognition/0/mnist_test_246.jpg"  #"D:/liudongbo/dataset/recognition/alh/MDEtMDEtMDAudHRm.png"
    mms = MinMaxScaler()
    for i in range(20):  # read 20 digits picture
        img = cv2.imread(Input_Numer[i], 0)  # Gray



        #if  img.shape != (28,28) :
        #    img.resize(28,28)



        print("img.shape =")
        print(img.shape)


        img_num[i] = img.copy()
        img = img.reshape(My_X.shape[1])
        My_X[i] = img.copy()




    My_test = mms.fit_transform(My_X)
    My_label_ohe = lenet5_train.encode_labels(My_Yd, 10)
    ##============================

    evaluate(My_test, My_label_ohe, My_Yd)


if __name__ == '__main__':
    main()

