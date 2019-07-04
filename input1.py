import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
train_dir_path = "E:/test/eyeclt/0/test/"
def getFile(fileDir):
    close = []
    lableclose = []
    open = []
    lableopen = []
    for file in os.listdir(fileDir):

        name = file.split(sep='.')
        if '0' == name[0]:
            close.append(fileDir + file)
            lableclose.append(0)
        else:
            #if '1' == name[0]:
            open.append(fileDir + file)
            lableopen.append(1)
        imageList = np.hstack((close,open))
        labelList = np.hstack((lableclose, lableopen))

    print("there are %d close\nthere %d open" % (len(close),len(open)))

    #imageList = np.hstack((cats,dogs))
    #labelList = np.hstack((lableCats,lableDogs))

    temp = np.array([imageList,labelList])
    temp = temp.transpose()
    np.random.shuffle(temp)

    imageList = list(temp[:,0])
    labelList = list(temp[:,1])
    labelList = [int(i) for i in labelList]

    return imageList,labelList


def getBatch(img,lable,img_w,img_h,batchSize,capacity):
    img = tf.cast(img,tf.string)

    lable = tf.cast(lable,tf.int32)

    inputQueue = tf.train.slice_input_producer([img,lable])
    lable = inputQueue[1]
    imgConents = tf.read_file(inputQueue[0])
    #lable = inputQueue[1]
    img = tf.image.decode_jpeg(imgConents,channels=3)
    img = tf.image.resize_image_with_crop_or_pad(img,img_w,img_h)
    img = tf.image.per_image_standardization(img)

    #img = tf.image.resize_images(img,[img_h,img_w],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.cast(img,tf.float32)

    imgBatch,lableBatch = tf.train.batch([img,lable],
                                         batch_size=batchSize,
                                         num_threads=64,
                                         capacity=capacity
                                         )

    lableBatch = tf.reshape(lableBatch,[batchSize])

    return imgBatch,lableBatch
getFile(train_dir_path)