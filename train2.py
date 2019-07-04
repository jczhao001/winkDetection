import os
import numpy as np
import tensorflow as tf
import input1
import modle1


N_CLASSES = 2
IMG_H = 80
IMG_W = 80
BATCH_SIZE = 40
CAPACITY = 1000
MAX_STEP = 2500
learning_rate = 0.00008

def training():
    train_dir_path = "E:/test/eyeclt/0/test/"
    logs_train_dir_path = "E:/test/eyelid/data/logs/"

    train,train_label = input1.getFile(train_dir_path)
    train_batch,train_label_batch = input1.getBatch(train,
                                        train_label,
                                        IMG_W,
                                        IMG_H,
                                        BATCH_SIZE,
                                        CAPACITY
                                        )
    train_logits = modle1.inference(train_batch,BATCH_SIZE,N_CLASSES)
    train_loss = modle1.losses(train_logits,train_label_batch)
    train_op = modle1.train(train_loss,learning_rate)
    train_acc = modle1.evalution(train_logits,train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir_path,sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _,tra_loss,tra_cc = sess.run([train_op,train_loss,train_acc])
            if step %100 == 0:
                print("step %d ,train loss = %.2f ,train acy = %.2f" % (step,tra_loss,tra_cc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str,step)
            if step % 1000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir_path,"model.ckpt")
                saver.save(sess,checkpoint_path,global_step=step)
    except tf.errors.OutOfRangeError:
        print("Done--limit reached.")
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

training()