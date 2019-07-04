from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import modle1
import os
import time


def get_image(test):
    file = os.listdir(test)
    n = len(file)
    ind = np.random.randint(0,n)
    imgdir = os.path.join(test,file[ind])

    image = Image.open(imgdir)
    plt.imshow(image)
    plt.show()
    image = image.resize([80,80])
    image = np.array(image)

    return image
def evalu_image():
    train_dir_path = "E:/test/eyeclt/2/test/"

    image_array = get_image(train_dir_path)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        image = tf.cast(image_array,tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image,[1,80,80,3])
        logit = modle1.inference(image,BATCH_SIZE,N_CLASSES)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32,shape=[80,80,3])

        logs_train_dir = "E:/test/eyelid/data/logs/"
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("reading...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
              globals_step = ckpt.model_checkpoint_path .split("/")[-1].split("-")[-1]
              saver.restore(sess,ckpt.model_checkpoint_path)
              print("loading success,global_step %s " % globals_step)
            else:
                print("No found.")
            start = time.time()
            prediction = sess.run(logit,feed_dict={x:image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print("this is close: %.6f "% prediction[:,0])
            else:
                print("this is a open: %.6f" % prediction[:,1])
            end = time.time()
            print("time:%.8f" % (end - start))
evalu_image()
