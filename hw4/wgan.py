import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import time
import os
from model import Generator, Discriminator, Discriminator_wgan_gp
import util3 as util

class Improved_WGAN(object):
    
    def __init__(self, data, FLAGS):
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.data = data
        self.FLAGS = FLAGS
        self.img_row = self.data.img_feat.shape[1]
        self.img_col = self.data.img_feat.shape[2]
        self.alpha = 10.
        self.d_epoch = 1
        self.gen_path()
        self.lambd = 0.25

        self.logs_path = './logs'
        self.log = True

        if self.log:
            self.summary_writer = tf.summary.FileWriter(self.logs_path,graph=tf.get_default_graph())
        


    def gen_path(self):
        # Output directory for models and summaries
        timestamp = str(time.strftime('%b-%d-%Y-%H-%M-%S'))
        self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", timestamp))
        print ("Writing to {}\n".format(self.out_dir))
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def build_model(self):

        self.g_net = Generator(   
                        embedding_size=self.FLAGS.embedding_dim, 
                        hidden_size=self.FLAGS.hidden,
                        img_row=self.img_row,
                        img_col=self.img_col)
        # self.d_net = Discriminator_wgan_gp( 
        #                 embedding_size=self.FLAGS.embedding_dim, 
        #                 hidden_size=self.FLAGS.hidden,
        #                 img_row=self.img_row,
        #                 img_col=self.img_col)

        self.d_net = Discriminator( 
                        embedding_size=self.FLAGS.embedding_dim, 
                        hidden_size=self.FLAGS.hidden,
                        img_row=self.img_row,
                        img_col=self.img_col)

        self.seq = tf.placeholder(tf.float32, [None, len(self.data.eyes_idx)+len(self.data.hair_idx)], name="seq")
        self.img = tf.placeholder(tf.float32, [64, self.img_row, self.img_col, 3], name="img")
        
    
        self.img_p = tf.placeholder(tf.float32, [64, self.img_row, self.img_col, 3], name="img_p")
        
        
        self.z = tf.placeholder(tf.float32, [None, self.FLAGS.z_dim])

        self.w_seq = tf.placeholder(tf.float32, [None, len(self.data.eyes_idx)+len(self.data.hair_idx)], name="w_seq")
        self.w_img = tf.placeholder(tf.float32, [None, self.img_row, self.img_col, 3], name="w_img")
        self.w_img_p = tf.placeholder(tf.float32, [64, self.img_row, self.img_col, 3], name="w_img_p")


        r_img, r_seq = self.img, self.seq

        self.f_img = self.g_net(r_seq, self.z)
        
        self.sampler = tf.identity(self.g_net(r_seq, self.z, reuse=True, train=False), name='sampler') 

        self.img_feats = (self.sampler + 1.)/2 * 255
        self.summaries = tf.summary.image('name', self.img_feats,  max_outputs=5)

        # TODO 
        """
        
            r img, r text -> 1
            f img, r text -> 0
            r img, w text -> 0
            w img, r text -> 0
        """
        self.d = self.d_net(r_seq, r_img, reuse=False)     # r img, r text
        self.d_1 = self.d_net(r_seq, self.f_img)         # f img, r text
        self.d_2 = self.d_net(self.w_seq, self.img)        # r img, w text
        self.d_3 = self.d_net(r_seq, self.w_img)        # w img, r text

        def gradient_penalty(real, fake, seq, f):
            def interpolate(a, b):
                shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
                alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
                inter = a + alpha * (b - a)
                inter.set_shape(a.get_shape().as_list())
                return inter

            x = interpolate(real, fake)
            pred = f(seq, x)
            print(x)
            gradients = tf.gradients(pred, x)[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
            gp = tf.reduce_mean((slopes - 1.)**2)
            return gp

        wd = tf.reduce_mean(self.d) - ( tf.reduce_mean(self.d_1) + tf.reduce_mean(self.d_2) + tf.reduce_mean(self.d_3))/3.
        gp = gradient_penalty(r_img, self.f_img, r_seq, self.d_net)
        self.d_loss = -wd + gp * 10.0

        self.g_loss = -tf.reduce_mean(self.d_1)




        # dcgan
        # self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_1, labels=tf.ones_like(self.d_1))) 

        # self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d, labels=tf.ones_like(self.d))) \
        #             + (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_1, labels=tf.zeros_like(self.d_1))) + \
        #                tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_2, labels=tf.zeros_like(self.d_2))) +\
        #                tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_3, labels=tf.zeros_like(self.d_3))) ) / 3 
        


        # alpha = tf.random_uniform(shape=self.img.get_shape(), minval=0.,maxval=1.)

        # differences = self.img_p - r_img  # This is different from WGAN-GP
        
        # interpolates = r_img + (alpha * differences)
        # D_inter=self.d_net(r_seq, interpolates) 
        # gradients = tf.gradients(D_inter, [interpolates])[0]
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        # gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)





        # differences_w = self.w_img_p - self.w_img  # This is different from WGAN-GP
        # interpolates_w = self.w_img + (alpha * differences_w)
        # D_inter_w=self.d_net(self.w_seq, interpolates_w) 

        # gradients = tf.gradients(D_inter_w, [interpolates_w])[0] + tf.gradients(D_inter, [interpolates])[0]
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        # gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        
        # self.d_loss += self.lambd * gradient_penalty




        self.global_step = tf.Variable(0, name='g_global_step', trainable=False)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_updates = tf.train.AdamOptimizer(self.FLAGS.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=self.d_net.vars)
            self.g_updates = tf.train.AdamOptimizer(self.FLAGS.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=self.g_net.vars, global_step=self.global_step)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())

    def get_perturbed_batch(self, minibatch):
        return minibatch + 0.5 * minibatch.std() * np.random.random(minibatch.shape)


    def train(self):
        batch_num = self.data.length//self.FLAGS.batch_size if self.data.length%self.FLAGS.batch_size==0 else self.data.length//self.FLAGS.batch_size + 1
        
        print("Start training WGAN...\n")

        for t in range(self.FLAGS.iter):

            d_cost = 0
            g_coat = 0

            for d_ep in range(self.d_epoch):

                img, tags,w_img, w_tags = self.data.next_data_batch(self.FLAGS.batch_size)
                z = self.data.next_noise_batch(len(tags), self.FLAGS.z_dim)

                # feed_dict = {
                #     self.seq:tags,
                #     self.img:img,
                #     self.img_p: self.get_perturbed_batch(img),
                #     self.z:z,
                #     self.w_img_p: self.get_perturbed_batch(w_img),
                #     self.w_seq:w_tags,
                #     self.w_img:w_img
                # }

                feed_dict = {
                    self.seq:tags,
                    self.img:img,
                    self.z:z,
                    self.w_seq:w_tags,
                    self.w_img:w_img
                }


                _, loss = self.sess.run([self.d_updates, self.d_loss], feed_dict=feed_dict)

                d_cost += loss/self.d_epoch

            z = self.data.next_noise_batch(len(tags), self.FLAGS.z_dim)
            
            # feed_dict = {
            #     self.img:img,
            #     self.img_p: self.get_perturbed_batch(img),
            #     self.w_seq:w_tags,
            #     self.w_img:w_img,
            #     self.w_img_p: self.get_perturbed_batch(w_img),
            #     self.seq:tags,
            #     self.z:z
            # }
            feed_dict = {
                self.img:img,
                self.w_seq:w_tags,
                self.w_img:w_img,
                self.seq:tags,
                self.z:z
            }

            _, loss, step = self.sess.run([self.g_updates, self.g_loss, self.global_step], feed_dict=feed_dict)

            current_step = tf.train.global_step(self.sess, self.global_step)

            g_cost = loss

            if current_step % self.FLAGS.display_every == 0:
                print("Epoch {}, Current_step {}".format(self.data.epoch, current_step))
                print("Discriminator loss :{}".format(d_cost))
                print("Generator loss     :{}".format(g_cost))
                print("---------------------------------")

            if current_step % self.FLAGS.checkpoint_every == 0:
                path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
                print ("\nSaved model checkpoint to {}\n".format(path))

            if current_step % self.FLAGS.dump_every == 0:


                self.eval(current_step)
                print("Dump test image")

    def eval(self, iters):
        
        #z = self.data.fixed_z
        z = self.data.next_noise_batch(len(self.data.test_tags_idx), self.FLAGS.z_dim)
        
        feed_dict = {
            self.seq:self.data.test_tags_idx,
            self.z:z
        }
       
        if self.log:
             summ = self.sess.run(self.summaries,feed_dict)
             self.summary_writer.add_summary(summ, iters)
        else:
            f_imgs = self.sess.run(self.sampler, feed_dict=feed_dict)
            util.dump_img(self.FLAGS.img_dir, f_imgs, iters)


