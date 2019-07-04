# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# import modle1
# import os
# import time
# import cv2
#
#
#
# def get_image(test):
#     # file = os.listdir(test)
#     # n = len(file)
#     # ind = np.random.randint(0,n)
#     # imgdir = os.path.join(test,file[ind])
#
#     # image = Image.open(test)
#     #image1= np.array(image)
#     #image12= cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
#     test = np.array(test)
#     plt.imshow(test)
#     plt.show()
#     # image = test.resize([80,80,3])
#     # image = np.array(image)
#     print(test)
#
#     return test
# def evalu_image():
#     train_dir_path = "E:/test/eyeclt/2/test/eye.png"
#     eye_haar = cv2.CascadeClassifier("E:/test/haar/haarcascade_righteye_2splits.xml")
#     # cam = cv2.VideoCapture(0)
#     # while True:
#
#     text = 'ori'
#     # ret, img = cam.read()
#     img = get_image(train_dir_path)
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     image_array = eye_haar.detectMultiScale(gray_img, 1.2, 5)
#
#     for eye_x, eye_y, eye_w, eye_h in image_array:
#         cut_eyeimg = img[eye_y:eye_y + eye_h, eye_x:eye_x + eye_w]
#         # cut_eyeimg = cv2.imread(train_dir_path)
#         org = (eye_y, eye_x)
#         cut_eyeimg = cv2.resize(cut_eyeimg, (80, 80), interpolation=cv2.INTER_CUBIC)
#         # cv2.imwrite('eye.png',cut_eyeimg)
#         cut_eyeimg = cv2.cvtColor(cut_eyeimg, cv2.COLOR_BGR2RGB)
#         # image_array = get_image(cut_eyeimg)
#         # cv2.imshow('img', cut_eyeimg)
#         # plt.imshow(cut_eyeimg)
#         # plt.show()
#         image_array = np.array(cut_eyeimg)
#         cv2.rectangle(img, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 255, 0), 2)
#
#
#     #     if cv2.waitKey(1) & 0xFF == ord('q'):
#     #         break
#     #
#     # cam.release()
#     # cv2.destroyAllWindows()
#
#         with tf.Graph().as_default():
#             BATCH_SIZE = 1
#             N_CLASSES = 2
#             image = tf.cast(image_array, tf.float32)
#             image = tf.image.per_image_standardization(image)
#             image = tf.reshape(image, [1, 80, 80, 3])
#             logit = modle1.inference(image,BATCH_SIZE, N_CLASSES)
#             logit = tf.nn.softmax(logit)
#
#             x = tf.placeholder(tf.float32, shape=[80, 80, 3])
#
#             logs_train_dir = "E:/test/eyelid/data/logs/"
#             saver = tf.train.Saver()
#
#             with tf.Session() as sess:
#                 print("reading...")
#                 ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#                 if ckpt and ckpt.model_checkpoint_path:
#                   globals_step = ckpt.model_checkpoint_path .split("/")[-1].split("-")[-1]
#                   saver.restore(sess,ckpt.model_checkpoint_path)
#                   print("loading success,global_step %s " % globals_step)
#                 else:
#                     print("No found.")
#                 start = time.clock()
#                 prediction = sess.run(logit, feed_dict={x: image_array})
#                 max_index = np.argmax(prediction)
#                 print(prediction)
#                 if max_index == 0:
#                     print("this is close_eye: %.6f "% prediction[:,0])
#                     text = 'Close'
#                 elif max_index == 1:
#                     print("this is a open_eye: %.6f " % prediction[:,1])
#                     text = 'Open'
#                 else:
#                     print('Unknown')
#                 end = time.clock()
#                 print("time:%.8f" % (end - start))
#
#             # org = (240, 80)
#             fontFace = cv2.FONT_HERSHEY_COMPLEX
#             fontScale = 1
#             fontcolor = (0, 255, 0)  # BGR
#             thickness = 1
#             lineType = 4
#             bottomLeftOrigin = 1
#             cv2.putText(img, text, org, fontFace, fontScale, fontcolor, thickness, lineType)
#             cv2.imshow('img', img)
#             cv2.waitKey(10)
#
# evalu_image()
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
