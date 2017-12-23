import pickle 
import os 
from util3 import Data
import util3 as util
import tensorflow as tf
import numpy as np
import sys
import os

#from Improved_ import WGAN
from wgan import Improved_WGAN

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of word embedding (default: 300)")
tf.flags.DEFINE_integer("hidden", 128, "hidden dimension of RNN hidden size")
tf.flags.DEFINE_integer("iter", 1000000, "number of training iter")
tf.flags.DEFINE_integer("z_dim", 100, "noise dimension")
tf.flags.DEFINE_integer("batch_size", 64, "batch size per iteration")
tf.flags.DEFINE_integer("display_every", 20, "predict model on dev set after this many steps (default: 200)")
tf.flags.DEFINE_integer("dump_every", 500, "predict model on dev set after this many steps (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 200)")

tf.flags.DEFINE_float("lr", 2e-4, "training learning rate")

tf.flags.DEFINE_string("img_dir", "./test_img/", "test image directory")
tf.flags.DEFINE_string("train_dir", "data\\faces", "training data directory")
tf.flags.DEFINE_string("tag_path", "./data/tags_clean.csv", "training data tags")
tf.flags.DEFINE_string("test_path", "./data/sample_testing_text.txt", "sample test format")
tf.flags.DEFINE_string("checkpoint_file", "", "checkpoint_file to be load")
tf.flags.DEFINE_string("prepro_dir", "./prepro/", "tokenized train data's path")
tf.flags.DEFINE_string("model", "Improved_WGAN", "init model name")

tf.flags.DEFINE_boolean("prepro", True, "preprocessing the training data")

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


PRE_DIR = 'prepro'
IMG_DIR = '../../data/faces'
TAG_PATH = '../../data/tags_clean.csv'
TEST_PATH = '../../data/sample_testing_text.txt'
if __name__ == '__main__':

    #img_feat, attrib_tags = util.preprocessing(PRE_DIR, IMG_DIR, TAG_PATH, eye_map, hair_map)

    img_feat = pickle.load(open(os.path.join(PRE_DIR, "img_feat.dat"), 'rb'))
    tags = pickle.load(open(os.path.join(PRE_DIR, "tags.dat"), 'rb'))


    img_feat = np.array(img_feat, dtype='float32')/127.5 - 1.

    data = Data(img_feat, tags, FLAGS.z_dim)
    data.load_test(TEST_PATH, hair_map, eye_map)

    Model = getattr(sys.modules[__name__], FLAGS.model)	

    print(Model)
    model = Model(data, FLAGS)
    model.build_model()
    model.train()






