from Utils.map_reader import phone_map_reader, phone_char_reader
from Utils.load_data import load_batched_data, phone_int_mapping
import tensorflow as tf
from dotdict import dotdict
import pickle

path_to_phone_map = './data/phones/48_39.map'
path_to_phone_char_map = './data/48phone_char.map'

mfcc_path = './data/mfcc'
label_path = './data/label'

mapping = phone_int_mapping(path_to_phone_char_map)

with open('./data/train_label.pkl', 'rb') as f:
    label = pickle.load(f)

labellist = []
input = []
for i in label:
    input = []
    for l in i:
        input.append(mapping[l])
    labellist.append(input)

with open('train_mapped_label.pkl', 'wb') as train_label:
    pickle.dump( labellist, train_label) 
        



# batchedData, maxTimeSteps = load_batched_data(mfcc_path, label_path, 1, 'train')
# print(batchedData)
# print(maxTimeSteps)

mode = 'train'
batch_size = 1
num_feature = 39
num_classes = 39
rnn
activation_fn = 'prelu'
optimizer_fn = 'adam'
cell_fn = 'lstm'

lr = 0.001
keep_prob = 0.8
grad_clip = 0.5

train_mfcc_dir = mfcc_path
train_label_dir = label_path




if __name__ == '__main__':
  runner = Runner()
  runner.run()