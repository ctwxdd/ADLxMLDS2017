import tensorflow as tf
import tensorflow.contrib as tc
import math
from libs.ops import *


class Generator(object):
    
    def __init__(self, 
        embedding_size, 
        hidden_size, 
        img_row, 
        img_col):
        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.img_row = img_row
        self.img_col = img_col
        
    def __call__(self, seq_idx, z, reuse=False, train=True):

        batch_size = tf.shape(seq_idx)[0]

        tags_vectors = seq_idx

        with tf.variable_scope("g_net") as scope:

            if reuse:
                scope.reuse_variables()

            emb = tc.layers.fully_connected(
                tags_vectors, 16,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            emb = lrelu(emb)


            noise_vector = tf.concat([emb, z], axis=1)

            fc2 = tc.layers.fully_connected(
                noise_vector, 4*4*256,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            fc2 = tf.layers.batch_normalization(fc2, training=train)
            fc2 = tf.reshape(fc2, [-1, 4, 4, 256])
            fc2 = tf.nn.relu(fc2)

            conv1 = tc.layers.convolution2d_transpose(
                fc2, 128, [5, 5], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            conv1 = tf.layers.batch_normalization(conv1, training=train)
            conv1 = tf.nn.relu(conv1)

            conv2 = tc.layers.convolution2d_transpose(
                conv1, 64, [5, 5], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            conv2 = tf.layers.batch_normalization(conv2, training=train)
            conv2 = tf.nn.relu(conv2)

            conv3 = tc.layers.convolution2d_transpose(
                conv2, 32, [5, 5], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            conv3 = tf.layers.batch_normalization(conv3, training=train)
            conv3 = tf.nn.relu(conv3)

            conv4 = tc.layers.convolution2d_transpose(
                conv3, 3, [5, 5], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            conv4 = tf.nn.tanh(conv4)

            return conv4

    @property
    def vars(self):
        return [var for var in tf.global_variables() if "g_net" in var.name]


class Discriminator_sn(object):
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
        self.c_dim = 3
    def __call__(self, seq_idx, img, reuse=True):

        batch_size = tf.shape(seq_idx)[0]

        tags_vectors = seq_idx
        hidden_activation = lrelu

        with tf.variable_scope("d_net") as scope:


            update_collection = None

            if reuse == True:
                update_collection = "NO_OPS"
                scope.reuse_variables()

            img = tf.reshape(img, [64,64,64,3])
 
            c0_0 = hidden_activation(conv2d(   img,  64, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c0_0'))
            c0_1 = hidden_activation(conv2d(c0_0, 128, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c0_1'))
            c1_0 = hidden_activation(conv2d(c0_1, 128, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c1_0'))
            c1_1 = hidden_activation(conv2d(c1_0, 256, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c1_1'))
            c2_0 = hidden_activation(conv2d(c1_1, 256, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c2_0'))

            c2_1 = hidden_activation(conv2d(c2_0, 512, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c2_1'))

            emb = tc.layers.fully_connected(
                tags_vectors, 16,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            emb = lrelu(emb)

            tags_vectors = tf.expand_dims(tf.expand_dims(tags_vectors, 1), 2)
            tags_vectors = tf.tile(tags_vectors, [1, 8, 8, 1])

            condition_info = tf.concat([c2_1, tags_vectors], axis=-1)
            c2_1 = hidden_activation(conv2d( condition_info, 512, 1, 1, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c2_2'))

            c3_0 = hidden_activation(conv2d(c2_1, 512, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c3_0'))
            c3_0 = tf.reshape(c3_0, [self.batch_size, -1])
            out_logit = linear(c3_0, self.c_dim, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='l4')
            out = tf.nn.sigmoid(out_logit)

        return out_logit

    @property
    def vars(self):
        return [var for var in tf.global_variables() if "d_net" in var.name]

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

    def __call__(self, tags_vectors, z, reuse=False, train=True):
        s = self.image_size # output image size [64]
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        gf_dim = 128
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

            emb = lrelu(emb)


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
            net = lrelu(net)

            net = tc.layers.convolution2d(
                net, gf_dim * 2, [3, 3], [1, 1],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net = tf.layers.batch_normalization(net, training=True)
            net = lrelu(net)


            net = tc.layers.convolution2d(
                net, gf_dim * 8, [3, 3], [1, 1],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),  #g_h1_res/conv2d3
                activation_fn=None
                )
            net = tf.layers.batch_normalization(net, training=True)
            #net = lrelu(net)

            net_h1 = tf.add(net_h0, net)
            net_h1 = lrelu(net_h1)


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
            net = lrelu(net)

            net = tc.layers.convolution2d(
                net, gf_dim, [3, 3], [1, 1],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net = tf.layers.batch_normalization(net, training=True)            
            net = lrelu(net)

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
            net_h3 = lrelu(net_h3)


            net_h4 = tc.layers.convolution2d_transpose(
                net_h3,  gf_dim * 2, [4, 4], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net_h4 = tf.layers.batch_normalization(net_h4, training=train)
            net_h4 = lrelu(net_h4)

            net_h5 = tc.layers.convolution2d_transpose(
                net_h4,gf_dim * 2, [4, 4], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net_h5 = tf.layers.batch_normalization(net_h5, training=train)
            net_h5 = lrelu(net_h5)

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

class Discriminator_sn_resnet(object):

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
        self.c_dim = 3
    def __call__(self, seq_idx, img, reuse=True):

        batch_size = tf.shape(seq_idx)[0]

        tags_vectors = seq_idx
        hidden_activation = lrelu

        s = 64 # output image size [64]
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        with tf.variable_scope("d_net") as scope:


            update_collection = None


            if reuse == True:
                update_collection = "NO_OPS"
                scope.reuse_variables()

            img = tf.reshape(img, [64,64,64,3])
 
            net_h0 = hidden_activation(conv2d(   img,  64, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='h_0'))
            net_h1 = hidden_activation(conv2d(net_h0, 128, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='h_1'))
            net_h2 = hidden_activation(conv2d(net_h1, 256, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='h_2'))
            net_h3 = conv2d(net_h2, 512, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='h_3')
            
            net = hidden_activation(conv2d(net_h3, 128, 1, 1, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='net_1',  padding="VALID"))
            net = hidden_activation(conv2d(net, 128, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='net_2'))
            net = hidden_activation(conv2d(net, 512, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='net_3'))

            net_h4 = tf.add(net_h3, net)
            net_h4 = lrelu(net_h4)

            tags_vectors = tc.layers.fully_connected(
                tags_vectors, 16,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            tags_vectors = lrelu(tags_vectors)


            tags_vectors = tf.expand_dims(tf.expand_dims(tags_vectors, 1), 2)
            tags_vectors = tf.tile(tags_vectors, [1, 4, 4, 1])

            net_h4_concat = tf.concat([net_h4, tags_vectors], axis=-1)

            net_h4 = hidden_activation(conv2d(net_h4_concat, 512, 1, 1, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02,  padding="VALID", name='h_4'))

            net_ho = tc.layers.convolution2d(
                net_h4, 1, [4, 4], [4, 4],
                padding='valid',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )

        return net_ho

    @property
    def vars(self):
        return [var for var in tf.global_variables() if "d_net" in var.name]


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
            
            net_h0 = lrelu(net_h0)


            net_h1 = tc.layers.convolution2d(
                net_h0, df_dim * 2, [4, 4], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            
            net_h1 = tf.layers.batch_normalization(net_h1, training=True)
            net_h1 = lrelu(net_h1)


            net_h2 = tc.layers.convolution2d(
                net_h1, df_dim * 4, [4, 4], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            
            net_h2 = tf.layers.batch_normalization(net_h2, training=True)
            net_h2 = lrelu(net_h2)

            net_h3 = tc.layers.convolution2d(
                net_h2, df_dim * 8, [4, 4], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            
            net_h3 = tf.layers.batch_normalization(net_h3, training=True)
            #net_h3 = lrelu(net_h3)

            net = tc.layers.convolution2d(
                net_h3, df_dim * 2, [1, 1], [1, 1],
                padding='valid',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net = tf.layers.batch_normalization(net, training=True)
            net = lrelu(net)

            net = tc.layers.convolution2d(
                net, df_dim * 2, [3, 3], [1, 1],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net = tf.layers.batch_normalization(net, training=True)
            net = lrelu(net)

            net = tc.layers.convolution2d(
                net, df_dim * 8, [3, 3], [1, 1],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net = tf.layers.batch_normalization(net, training=True)
            
            net_h4 = tf.add(net_h3, net)
            net_h4 = lrelu(net_h4)


            tags_vectors = tc.layers.fully_connected(
                tags_vectors, 16,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            tags_vectors = lrelu(tags_vectors)


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
            net_h4 = lrelu(net_h4)

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



    

    def generator(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        hidden_activation = tf.nn.relu
        output_activation = tf.nn.tanh
        with tf.variable_scope("generator", reuse=reuse) as vs:
            l0  = hidden_activation(batch_norm(linear(z, 4 * 4 * 512, name='l0', stddev=0.02), name='bn0', is_training=is_training))
            l0  = tf.reshape(l0, [self.batch_size, 4, 4, 512])
            dc1 = hidden_activation(batch_norm(deconv2d( l0, [self.batch_size,  8,  8, 256], name='dc1', stddev=0.02), name='bn1', is_training=is_training))
            dc2 = hidden_activation(batch_norm(deconv2d(dc1, [self.batch_size, 16, 16, 128], name='dc2', stddev=0.02), name='bn2', is_training=is_training))
            dc3 = hidden_activation(batch_norm(deconv2d(dc2, [self.batch_size, 32, 32,  64], name='dc3', stddev=0.02), name='bn3', is_training=is_training))
            dc4 = hidden_activation(batch_norm(deconv2d(dc3, [self.batch_size, 64, 64,  32], name='dc4', stddev=0.02), name='bn4', is_training=is_training))
            dc5 = deconv2d(dc4, [self.batch_size, 64, 64, 3], 3, 3, 1, 1, name='dc5', stddev=0.02)
        variables = tf.contrib.framework.get_variables(vs)
        return dc5, variables


