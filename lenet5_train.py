#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thur July 5 19:50:49 2018

@author: Dongbo Liu
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import struct
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import cv2,csv
import lenet5_infernece
import readDataSet as rd
import time


def encode_labels( y, k):
    """Encode labels into one-hot representation
    """
    onehot = np.zeros((y.shape[0],k ))
    for idx, val in enumerate(y):
        onehot[idx,val] = 1.0  ##idx=0~xxxxx，if val =3 ,表示欄位3要設成1.0
    return onehot

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    if kind=='train':
        labels_path=os.path.abspath('D:\\liudongbo\\dataset\\MNIST\\train-labels.idx1-ubyte')
        images_path=os.path.abspath('D:\\liudongbo\\dataset\\MNIST\\train-images.idx3-ubyte')
    else:
        labels_path=os.path.abspath('D:\\liudongbo\\dataset\\MNIST\\t10k-labels.idx1-ubyte')
        images_path=os.path.abspath('D:\\liudongbo\\dataset\\MNIST\\t10k-images.idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99
TRAIN_DATA_DIR = "D:/liudongbo/dataset/train_set_02/"   #"D:/liudongbo/dataset/notMNIST_large/"
MODEL_SAVE_PATH = "D:/liudongbo/models/Mnist_models/"
MODEL_NAME = "lenet5_model"
INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
display_step = 100
learning_rate_flag=True
BATCHNUMBER = 1


def initVarible():
    shuffle = True
    batch_idx = 0
    train_acc=[]
    # 定義輸出為4維矩陣的placeholder
    x_ = tf.placeholder(tf.float32, [None, INPUT_NODE],name='x-input')
    x = tf.reshape(x_, shape=[-1, 28, 28, 1])

    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = lenet5_infernece.inference(x,True,regularizer,tf.AUTO_REUSE)  #tf.AUTO_REUSE
    global_step = tf.Variable(0, trainable=False)

    # Evaluate model
    pred_max=tf.argmax(y,1)
    y_max=tf.argmax(y_,1)
    correct_pred = tf.equal(pred_max,y_max)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

   # 定義損失函數、學習率、及訓練過程。

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))



    return  global_step, shuffle,batch_idx, loss, x, y_, accuracy, train_acc


def train(X_train,y_train_lable,global_step, shuffle, batch_idx, loss, x, y_, accuracy, train_acc,saver,sess):
   #with tf.Graph().as_default() as g:


    batch_len =int( X_train.shape[0]/BATCH_SIZE)
    #test_batch_len =int( X_test.shape[0]/BATCH_SIZE)
    test_acc=[]

    train_idx=np.random.permutation(batch_len)#打散btach_len=600 group


    if learning_rate_flag==True:
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            X_train.shape[0] / BATCH_SIZE, LEARNING_RATE_DECAY,
            staircase=True)
    else:
        learning_rate = 0.001 #Ashing test
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    pretrained_model = None
    files = os.listdir(MODEL_SAVE_PATH)
    print("len(files)=")
    print(len(files))
    if len(files) != 0:
        pretrained_model = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
        print('Pre-trained model: %s' % pretrained_model)

    print("pretrained_model=")
    print(pretrained_model)
    # Start running operations on the Graph.
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    #sess.run(tf.global_variables_initializer())


    #with tf.Session() as sess:
    #with sess.as_default():


    if pretrained_model != None:
        print('Restoring pretrained model: %s' % pretrained_model)
        # saver.restore(sess, pretrained_model)
        saver.restore(sess, pretrained_model)

    step = 1
    print ("Start  training!")
    while step	< TRAINING_STEPS:
        #batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        if shuffle==True:
            batch_shuffle_idx=train_idx[batch_idx]
            batch_xs=X_train[batch_shuffle_idx*BATCH_SIZE:batch_shuffle_idx*BATCH_SIZE+BATCH_SIZE]
            batch_ys=y_train_lable[batch_shuffle_idx*BATCH_SIZE:batch_shuffle_idx*BATCH_SIZE+BATCH_SIZE]
        else:
            batch_xs=X_train[batch_idx*BATCH_SIZE:batch_idx*BATCH_SIZE+BATCH_SIZE]
            batch_ys=y_train_lable[batch_idx*BATCH_SIZE:batch_idx*BATCH_SIZE+BATCH_SIZE]

        if batch_idx<batch_len:
            batch_idx+=1
            if batch_idx==batch_len:
                batch_idx=0
        else:
            batch_idx=0

        reshaped_xs = np.reshape(batch_xs, (
                BATCH_SIZE,
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS))


        # Fit training using batch data
        _, loss_value, total_step = sess.run([train_step, loss, global_step], feed_dict={x: reshaped_xs, y_: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: reshaped_xs, y_: batch_ys})
        train_acc.append(acc)
        if step % display_step == 0:
            print("After %d training step(s), loss on training batch is %g,Training Accuracy=%g" % (step, loss_value,acc))
            print("the training datasets is %dth batch"%(int(total_step/TRAINING_STEPS)+1))

        step += 1

    print ("Optimization Finished!")
    train_acc_avg=tf.reduce_mean(tf.cast(train_acc, tf.float32))
    print("Average Training Accuracy=",sess.run(train_acc_avg))
        #if dataFlag == 1 :
         #  print("Save model...")
           #saver.save(sess, "./lenet5/lenet5_model")
        #saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
        #return  sess


def main(argv=None):
    #mnist = input_data.read_data_sets("./mnist", one_hot=True)
    #### Loading the data
    X_train, y_train = load_mnist('..\mnist', kind='train')
    print('X_train Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1])) #X_train=60000x784
    X_test, y_test = load_mnist('mnist', kind='t10k')					 #X_test=10000x784
    print('X_test Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
    #image_datas, image_labels, nrof_images_total, removeCount  = rd.get_datas_and_labels("D:/liudongbo/dataset/train_set/")



   # test_image_datas, test_image_labels, test_nrof_images_total, test_removeCount = rd.get_datas_and_labels("D:/liudongbo/dataset/test_set/")





    print("train_data.shape=",X_train.shape)
    print("train_label.shape=",y_train.shape)
    print("train_label=",y_train)
    print("test_data.shape=",X_test.shape)

    start_time = time.time()
    train_image_path_list = rd.get_image_paths(TRAIN_DATA_DIR)
    duration = time.time() - start_time
    print('read paths Time %.3f\t' %(duration))
    train_time = 0
    read_data_time = 0

    totalPathNumbers = len(train_image_path_list)
    batchsize = int(totalPathNumbers/BATCHNUMBER) + 1



    with tf.Graph().as_default():

        global_step, shuffle,batch_idx, loss, x, y_, accuracy, train_acc = initVarible()
        # 初始化TensorFlow持久化類。
        saver = tf.train.Saver()

        with tf.Session() as sess:
           tf.global_variables_initializer().run()
           i = 0
           dataFlag = 0
           while i < totalPathNumbers :
              image_paths = []
              j = min(i+batchsize, totalPathNumbers)
              while i < j:
                  image_paths.append(train_image_path_list[i])
                  i = i+1
                  print("i=")
                  print(i)

              read_data_time_start = time.time()

              image_datas, image_labels = rd.get_images_and_labels(image_paths)

              read_data_time_duration = time.time() -read_data_time_start
              read_data_time += read_data_time_duration

              mms=MinMaxScaler()
    #X_train=mms.fit_transform(X_train)
    #X_test=mms.transform(X_test)

              X_train1 = mms.fit_transform(image_datas)
     #  X_test = mms.transform(test_image_datas)

    #stdsc=StandardScaler()
    #X_train=stdsc.fit_transform(X_train)
    #X_test=stdsc.transform(X_test)

    #y_train_lable = encode_labels(y_train,10)
    #y_test_lable = encode_labels(y_test,10)

              y_train_lable = image_labels
     #  y_test_lable = test_image_labels

              print("y_train_lable.shape=",y_train_lable.shape)
     #  print("y_test_lable.shape=",y_test_lable.shape)
    ##============================
              if j == totalPathNumbers:
               dataFlag = 1

              train_time_start = time.time()
              train(X_train1,y_train_lable,global_step, shuffle,batch_idx, loss, x, y_, accuracy, train_acc,saver,sess)
              duration_train_time  = time.time() - train_time_start
              train_time += duration_train_time


           saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
    print('read paths Time %.3f\t' % (duration))
    print('Train Time %.3f\t' %(train_time))
    print('Read data Time %.3f\t' %(read_data_time))


if __name__ == '__main__':
    main()


