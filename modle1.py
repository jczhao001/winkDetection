import tensorflow as tf

def inference(img,batchSize,n_classes):


    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3,3,3,16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,
                                                                              dtype=tf.float32
                                                                              )
                                  )
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1)
                                 )
        conv = tf.nn.conv2d(img,
                            weights,
                            strides=[1,1,1,1],
                            padding="SAME"
                            )
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation,name="conv1")

    with tf.variable_scope("pool_1") as scope:
        pool1 = tf.nn.max_pool(conv1,
                               ksize=[1,3,3,1],
                               strides=[1,2,2,1],
                               padding="SAME",
                               name="pool_1"
                               )
        norm1 = tf.nn.lrn(pool1,
                          depth_radius=4,
                          bias=1.0,
                          alpha=0.001/9.0,
                          beta=0.75,
                          name="norm1"
                          )


    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3,3,16,16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,
                                                                              dtype=tf.float32
                                                                              )
                                  )
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1)
                                 )

        conv = tf.nn.conv2d(norm1,
                            weights,
                            strides=[1, 1, 1, 1],
                            padding="SAME"
                            )
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation,name="conv2")

    with tf.variable_scope("pool_2") as scope:
        norm2 = tf.nn.lrn(conv2,
                          depth_radius=4,
                          bias=1.0,
                          alpha=0.001 / 9.0,
                          beta=0.75,
                          name="norm2"
                          )

        pool2 = tf.nn.max_pool(norm2,
                                ksize=[1, 3, 3, 1],
                                strides=[1, 2, 2, 1],
                                padding="SAME",
                                name="pool_2"
                                )
########################################################################
        #norm2 = tf.nn.lrn(pool2,
        #                    depth_radius=4,
        #                   bias=1.0,
        #                    alpha=0.001 / 9.0,
        #                    beta=0.75,
        #                    name="norm2"
        #                    )
##########################################################################
    with tf.variable_scope("full_1") as scope:
        reshape = tf.reshape(pool2,shape=[batchSize,-1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                              dtype=tf.float32
                                                                              )
                                  )
        biases = tf.get_variable("biases",
                                shape=[128],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1)
                                )
        full_1=tf.nn.relu(tf.matmul(reshape,weights) + biases,name="full_1")



    with tf.variable_scope("full_2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                              dtype=tf.float32
                                                                              )
                                  )
        biases = tf.get_variable("biases",
                                  shape=[128],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1)
                                  )
        full_2 = tf.nn.relu(tf.matmul(full_1, weights) + biases, name="full_2")

    with tf.variable_scope("softmax") as scope:


        weights = tf.get_variable("weights",
                                    shape=[128,n_classes],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                                dtype=tf.float32
                                                                                )
                                    )
        biases = tf.get_variable("biases",
                                    shape=[n_classes],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1)
                                    )
        softmax_lnr = tf.add(tf.matmul(full_2,weights),biases,name="softmax_lnr")
       # softmax_lnr = tf.nn.softmax(softmax_lnr)
    return softmax_lnr

def losses(logits,labels):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels,
                                                                       name="xet_per_exp"
                                                                       )
        loss = tf.reduce_mean(cross_entropy,name="loss")
        tf.summary.scalar(scope.name + "loss",loss)
    return loss
def train(loss,learnRate):
    with tf.name_scope("optimizer") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learnRate)
        global_step = tf.Variable(0,name="glo_stp",trainable=False)
        train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op
def evalution(logits,labels):
    with tf.variable_scope("ac_acy") as scope:
        correct = tf.nn.in_top_k(logits,labels,1)
        correct = tf.cast(correct,tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name +"ac_acy",accuracy)
    return accuracy
