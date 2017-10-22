import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def maskcost(output, target):

    output = tf.nn.log_softmax(output)
    cross_entropy = -tf.reduce_sum(target * output, 2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
    cross_entropy *= mask
    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy /= tf.reduce_sum(mask, 1)

    return tf.reduce_mean(cross_entropy, 0)

def mask_accuracy(output, target):

    mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
    #mask = tf.Print(mask, [mask], summarize=500 )
    correct_prediction = tf.cast(tf.equal(tf.argmax(output, 2), tf.argmax(target, 2)), tf.float32)
    correct_prediction *= mask
    #correct_prediction = tf.Print(correct_prediction, [correct_prediction], summarize=500 )
    correct_prediction = tf.reduce_sum(correct_prediction, 1)
    mask = tf.reduce_sum(mask, 1)
    return tf.reduce_mean(correct_prediction / mask)