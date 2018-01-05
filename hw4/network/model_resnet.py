import tensorflow as tf
import tensorflow.contrib as tc
import math
from libs.ops import *

class Generator_resnet(object):

    def __init__(self, 
        embedding_size, 
        hidden_size, 
        img_row, 
        img_col):
        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.img_row = img_row
        self.img_col = img_col
        self.batch_size = 64
        self.image_size = 64

    def __call__(self, tags_vectors, z, reuse=False, train=True, batch_size = 64):
        self.batch_size = batch_size
        s = self.image_size # output image size [64]
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        gf_dim = 64
        c_dim = 3

        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("g_net") as scope:

            if reuse:
                scope.reuse_variables()
            
            emb = tc.layers.fully_connected(
                tags_vectors, 16,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )

            emb =lrelu(emb)


            noise_vector = tf.concat([emb, z], axis=1)

            net_h0 = tc.layers.fully_connected(
                noise_vector, gf_dim*8*s16*s16,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net_h0 = tf.layers.batch_normalization(net_h0, training=train)
            net_h0 = tf.reshape(net_h0, [-1, s16, s16, gf_dim*8])
            #net_h0 = tf.nn.relu(net_h0)


            net = tc.layers.convolution2d(
                net_h0, gf_dim * 2, [1, 1], [1, 1],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net = tf.layers.batch_normalization(net, training=True)
            net =lrelu(net)

            net = tc.layers.convolution2d(
                net, gf_dim * 2, [3, 3], [1, 1],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net = tf.layers.batch_normalization(net, training=True)
            net =lrelu(net)


            net = tc.layers.convolution2d(
                net, gf_dim * 8, [3, 3], [1, 1],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),  #g_h1_res/conv2d3
                activation_fn=None
                )
            net = tf.layers.batch_normalization(net, training=True)
            #net =lrelu(net)

            net_h1 = tf.add(net_h0, net)
            net_h1 =lrelu(net_h1)


            net_h2 = tc.layers.convolution2d_transpose(
                net_h1, gf_dim * 4 , [4, 4], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net_h2 = tf.layers.batch_normalization(net_h2, training=train)

            net = tc.layers.convolution2d(
                net_h2, gf_dim, [1, 1], [1, 1],
                padding='valid',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
            )
            net = tf.layers.batch_normalization(net, training=True)
            net =lrelu(net)

            net = tc.layers.convolution2d(
                net, gf_dim, [3, 3], [1, 1],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net = tf.layers.batch_normalization(net, training=True)            
            net =lrelu(net)

            net = tc.layers.convolution2d(
                net, gf_dim * 4, [3, 3], [1, 1],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net = tf.layers.batch_normalization(net, training=True)
            

            net = tc.layers.convolution2d(
                net, gf_dim * 4, [3, 3], [1, 1],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )

            net = tf.layers.batch_normalization(net, training=True)

            net_h3 = tf.add(net_h2, net)
            net_h3 =lrelu(net_h3)


            net_h4 = tc.layers.convolution2d_transpose(
                net_h3,  gf_dim * 2, [4, 4], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net_h4 = tf.layers.batch_normalization(net_h4, training=train)
            net_h4 =lrelu(net_h4)

            net_h5 = tc.layers.convolution2d_transpose(
                net_h4,gf_dim * 2, [4, 4], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net_h5 = tf.layers.batch_normalization(net_h5, training=train)
            net_h5 =lrelu(net_h5)

            net_ho = tc.layers.convolution2d_transpose(
                net_h5, c_dim, [4, 4], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )

            net_ho = tf.nn.tanh(net_ho)

        return net_ho

    @property
    def vars(self):
        return [var for var in tf.global_variables() if "g_net" in var.name]




class Discriminator_resnet(object):
    def __init__(self, 
        embedding_size, 
        hidden_size,
        img_row,
        img_col):

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.img_row = img_row
        self.img_col = img_col
        self.image_size = 64

    
    def __call__(self, tags_vectors, img, reuse=True):


        s = self.image_size # output image size [64]
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        df_dim = 64
        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("d_net") as scope:

            if reuse:
                scope.reuse_variables()
            

            net_h0 = tc.layers.convolution2d(
                img, df_dim, [4, 4], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            
            net_h0 =lrelu(net_h0)


            net_h1 = tc.layers.convolution2d(
                net_h0, df_dim * 2, [4, 4], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            
            net_h1 = tf.layers.batch_normalization(net_h1, training=True)
            net_h1 =lrelu(net_h1)


            net_h2 = tc.layers.convolution2d(
                net_h1, df_dim * 4, [4, 4], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            
            net_h2 = tf.layers.batch_normalization(net_h2, training=True)
            net_h2 =lrelu(net_h2)

            net_h3 = tc.layers.convolution2d(
                net_h2, df_dim * 8, [4, 4], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            
            net_h3 = tf.layers.batch_normalization(net_h3, training=True)
            #net_h3 =lrelu(net_h3)

            net = tc.layers.convolution2d(
                net_h3, df_dim * 2, [1, 1], [1, 1],
                padding='valid',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net = tf.layers.batch_normalization(net, training=True)
            net =lrelu(net)

            net = tc.layers.convolution2d(
                net, df_dim * 2, [3, 3], [1, 1],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net = tf.layers.batch_normalization(net, training=True)
            net =lrelu(net)

            net = tc.layers.convolution2d(
                net, df_dim * 8, [3, 3], [1, 1],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net = tf.layers.batch_normalization(net, training=True)
            
            net_h4 = tf.add(net_h3, net)
            net_h4 =lrelu(net_h4)


            tags_vectors = tc.layers.fully_connected(
                tags_vectors, 16,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            tags_vectors =lrelu(tags_vectors)


            tags_vectors = tf.expand_dims(tf.expand_dims(tags_vectors, 1), 2)
            tags_vectors = tf.tile(tags_vectors, [1, 4, 4, 1])

            net_h4_concat = tf.concat([net_h4, tags_vectors], axis=-1)

            net_h4 = tc.layers.convolution2d(
                net_h4_concat, df_dim * 8, [1, 1], [1, 1],
                padding='valid',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net_h4 = tf.layers.batch_normalization(net_h4, training=True)
            net_h4 =lrelu(net_h4)

            net_ho = tc.layers.convolution2d(
                net_h4, 1, [s16, s16], [s16, s16],
                padding='valid',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )

            return net_ho

    @property
    def vars(self):
        return [var for var in tf.global_variables() if "d_net" in var.name]

