import pickle 
import os 
from util import Data
import util
import tensorflow as tf
import numpy as np
import sys
import os
import scipy.stats as stats
from scipy import misc

#from Improved_ import WGAN

from model import Generator

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

tf.flags.DEFINE_integer("z_dim", 100, "noise dimension")
tf.flags.DEFINE_string("img_dir", "./early/", "test image directory")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

hair_color = ['unk', 'orange hair', 'white hair', 'aqua hair', 'gray hair',
    'green hair', 'red hair', 'purple hair', 'pink hair',
    'blue hair', 'black hair', 'brown hair', 'blonde hair']

eye_color = ['unk','gray eyes', 'black eyes', 'orange eyes',
    'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
    'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

hair_map = {}
eye_map = {}

for idx, h in enumerate(hair_color):
    hair_map[h] = idx

for idx, e in enumerate(eye_color):
    eye_map[e] = idx

TEST_PATH = '../../data/sample_testing_text.txt'

def make_one_hot( hair, eye):

    eyes_hot = np.zeros([len(eye_color)])
    eyes_hot[eye] = 1
    hair_hot = np.zeros([len(hair_color)])
    hair_hot[hair] = 1
    tag_vec = np.concatenate((eyes_hot, hair_hot))

    return tag_vec


def load_test(test_path, hair_map, eye_map):

    test = []

    with open(test_path, 'r') as f:

        for line in f.readlines():
            hair = 0
            eye = 0
            if line == '\n':
                break
            line = line.strip().split(',')[1]
            p = line.split(' ')
            p1 = ' '.join(p[:2]).strip()
            p2 = ' '.join(p[-2:]).strip()
        
            if p1 in hair_map:
                hair = hair_map[p1]
            elif p2 in hair_map:
                hair = hair_map[p2]
            
            if p1 in eye_map:
                eye = eye_map[p1]
            elif p2 in eye_map:
                eye = eye_map[p2]

            test.append(make_one_hot(hair, eye))
    
    return test


def dump_img(img_dir, img_feats, test):

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_feats = (img_feats + 1.)/2 * 255.
    img_feats = np.array(img_feats, dtype=np.uint8)

    for idx, img_feat in enumerate(img_feats):
        path = os.path.join(img_dir, 'sample_{}_{}.jpg'.format(test, idx+1))
        misc.imsave(path, img_feat)



if __name__ == '__main__':

    seq = tf.placeholder(tf.float32, [None, len(hair_color)+len(eye_color)], name="seq")      
    z = tf.placeholder(tf.float32, [None, FLAGS.z_dim])

    g_net = Generator(  embedding_size=100, 
                        hidden_size=100,
                        img_row=64,
                        img_col=64)

    result = g_net(seq, z)


    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=None)
    saver.restore(sess, save_path='./ckpt/model-128000')

    z_sampler = stats.truncnorm((-1 - 0.) / 1., (1 - 0.) / 1., loc=0., scale=1)

    test = load_test(TEST_PATH, hair_map, eye_map)
    
    
    
    for idx, t in enumerate(test):
        z_noise = z_sampler.rvs([5, FLAGS.z_dim])

        t = np.expand_dims(t, axis=0)
        cond = np.repeat(t, 5, axis=0)
        feed_dict = {seq: cond,  z:z_noise}

        sampler = tf.identity(g_net(seq, z, reuse=True, train=False), name='sampler')

        f_imgs = sess.run(sampler, feed_dict=feed_dict)

        dump_img(FLAGS.img_dir, f_imgs, idx+1)
