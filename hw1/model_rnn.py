import tensorflow as tf
import numpy as np
import pickle
import time
import argparse
import os 
import sys
sys.path.append("./Utils")
from Utils import import_data
from Utils.layers import weight_variable, bias_variable, conv2d, maskcost, mask_accuracy
from Utils.preprocessing import removeNoise, remove_duplicate
from Utils import map_reader
from tensorflow.contrib.rnn import BasicLSTMCell

#choose which data to use

#data_folder = './data/mfcc'
#data_folder = './data/fbank'
data_folder = './data'

### Parameters
NUM_CLASSES = 48

model_name = 'rnn'
logdir = './tmp/' + model_name
num_epochs = 15
train_batch_size = 128
num_hidden = 64
num_lstm_layers = 1
use_dropout = True
optimizer_name='Adam'
learn_rate = 0.001
contex = 0

### Internal variables
num_features = 108
num_classes = 48
start_date = time.strftime("%d-%m-%y/%H.%M.%S")
save_loc = "./output"

x = tf.placeholder(tf.float32, [None, None, num_features], name='input')
y_ = tf.placeholder(tf.float32, [None, None, num_classes], name='y_')
sequence_len = tf.placeholder(tf.int32, [None], name='sequence_len')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

### Ops
sess = None
saver = None

def RNN(X):

    with tf.name_scope("inlayer"):

        X = tf.reshape(X, [-1, num_features])
        W_inlayer = weight_variable([num_features, num_hidden])
        b_inlayer = bias_variable([num_hidden])
        X_in = tf.matmul(X, W_inlayer) + b_inlayer

        X_in = tf.reshape(X_in, [train_batch_size, -1, num_hidden])

    with tf.name_scope("RNN_CELL"):

        cells = []

        for n in range(num_lstm_layers):
            cells.append(tf.contrib.rnn.BasicLSTMCell(num_hidden, state_is_tuple=True))

        lstm_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,input_keep_prob=keep_prob)
        rnn_out, states = tf.nn.dynamic_rnn(lstm_cell, X_in, sequence_length=sequence_len, dtype=tf.float32)

    with tf.name_scope("out_layer"):
        
        rnn_out = tf.reshape(rnn_out, [-1, num_hidden])
        W_outlayer = weight_variable([num_hidden, num_classes])
        b_outlayer = bias_variable([num_classes])

        results = tf.matmul(rnn_out, W_outlayer) + b_outlayer

        results = tf.reshape(results, [train_batch_size, -1, num_classes])
    
    print("Model creation done")
    
    return results

def train(train_set, eval_set, y, cost, optimizer):


    global merged

    with tf.name_scope('accuracy'):
        accuracy = mask_accuracy(y, y_)
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    
    for epoch in range(num_epochs):
        print("epoch %d:" % epoch)
        epoch_done = False

        batch = 0
        while not epoch_done:

            batch += 1
            batch_data, batch_labels, seq_len = train_set.next_batch(train_batch_size, _pad=contex)
            
            if batch_data is None:
                # Epoch is finished
                epoch_done = True
                break

            feed_dict = {
                x: batch_data,
                y_: batch_labels,
                sequence_len: seq_len,
                keep_prob: 1.0
            }                        
            
            if batch%100 == 0:

                summary, train_accuracy = sess.run([merged, accuracy],feed_dict=feed_dict)  
                print("epoch %d, batch %d, training accuracy %g"%(epoch, batch, train_accuracy))               
                train_writer.add_summary(summary, (epoch * 3696 / train_batch_size) + batch)

            feed_dict[keep_prob] = 0.5
            
            c_e, _ = sess.run([cost, optimizer], feed_dict=feed_dict)

            if batch%100 == 0:
                print("cross entropy: %g"%c_e)

        num_examples = train_set.data.shape[0]
        train_set.reset_epoch(train_batch_size)
        
        end_of_eval = False

        accuracies = []

        while not end_of_eval:

            batch_data, batch_labels, seq_len = train_set.next_batch(train_batch_size, _pad=contex)

            if batch_data is None:
                end_of_eval = True
                break

            feed_dict = {
                x: batch_data,
                y_: batch_labels,
                sequence_len: seq_len,
                keep_prob: 1.0
            }
            
            acc = sess.run([accuracy], feed_dict=feed_dict)
            accuracies.append(acc)

        train_set.reset_epoch(train_batch_size)
        end_of_cross_val = False
        accuracies_val = []
        idx = 0

        while not end_of_cross_val:
            idx+=1
            batch_data, batch_labels, seq_len = eval_set.next_batch(train_batch_size, _pad=contex)

            if batch_data is None:
                end_of_cross_val = True
                break

            feed_dict = {
                x: batch_data,
                y_: batch_labels,
                sequence_len: seq_len,
                keep_prob: 1.0
            }
            acc = sess.run([accuracy], feed_dict=feed_dict)
            accuracies_val.append(acc)

        eval_set.reset_epoch(train_batch_size)

        acc_mean = np.mean(np.array(accuracies))
        eval_acc_mean = np.mean(np.array(accuracies_val))
        print("epoch %d finished, accuracy: %g" % (epoch, acc_mean))
        print("valication accuracy: %g" % (eval_acc_mean))
        save_path = saver.save(sess, "%s/%s/%s/epoch%d_model.ckpt"%(save_loc, model_name, start_date, epoch))
        print("Model for epoch %d saved in file: %s"%(epoch, save_path))
        train_set.reset_epoch(train_batch_size)

    
def train_rnn(data_folder, model_file = None):
    y = RNN(x)
    print("Loading training pickles..")   

    train_set, eval_set = import_data.load_dataset(data_folder + '/train_data.pkl', 
                                         data_folder + '/train_label.pkl',
                                         batch_size=train_batch_size)
    print("Loading done")
    
    global sess
    global train_writer
    global saver
    global merged

    saver = tf.train.Saver()

    # Create the dir for the model
    if not os.path.isdir('%s/%s/%s'%(save_loc, model_name, start_date)):
        try:
            os.makedirs('%s/%s/%s'%(save_loc, model_name,start_date))
        except OSError:
            if not os.path.isdir('%s/%s/%s'%(save_loc, model_name, start_date)):
                raise
 
    cost = maskcost(y, y_)

    tf.summary.scalar('cross_entropy', cost)

    if optimizer_name == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate).minimize(cost)
    elif optimizer_name == 'RmsProp':
        optimizer = tf.train.RMSPropOptimizer(learn_rate).minimize(cost)
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        train_writer = tf.summary.FileWriter(logdir, sess.graph)
        #test_writer = tf.summary.FileWriter(logdir)
    
        if model_file:
            saver.restore(sess, model_file)
            print("Model restored")
        else:
            sess.run(init)
        
        print("Training network. Date: %s" % start_date)
        train(train_set, eval_set, y, cost, optimizer)
        
        save_path = saver.save(sess, "%s/%s/%s/model.ckpt"%(save_loc, model_name, start_date))
        print("Model saved in file: %s" % save_path)
    
def evaluate_rnn_model_from_file(data_folder, model_file):
    #Load the CNN_RNN graph
    y = RNN(x)

    saver = tf.train.Saver()
    global sess

    with tf.Session() as sess:
        saver.restore(sess, model_file)
        print("Model restored")
        evaluate_rnn(data_folder, y)

        
def evaluate_rnn(data_folder, y):

    phone_list = map_reader.phone_list('./data/48phone_char.map')
    phone_char_map = map_reader.phone_char_reader('./data/48phone_char.map')
    phone_map = map_reader.phone_map_reader('./data/phones/48_39.map')
    print("Loaded phone mapping")
    
    r = tf.argmax(y,2)
    test_set = import_data.load_test_dataset(data_folder + '/test_data.pkl', data_folder + '/test_name.pkl', batch_size=1)

    end_of_test = False

    f = open(csv_file_name, 'w')
    f.write('id,phone_sequence\n')

    while not end_of_test:

        batch_data, seq_len, name_list = test_set.next_test_batch(16, _pad=contex)

        if batch_data is None:
            end_of_test = True
            break

        feed_dict = {
            x: batch_data,
            sequence_len: seq_len,
            keep_prob: 1.0
        }
        
        results = sess.run([r], feed_dict=feed_dict)[0]

        for result, l, name in zip(results, seq_len, name_list):

            mapped_result = [phone_list[i] for i in result[:l]]
            mapped_result = removeNoise(mapped_result)
            phone39_result = [phone_map[i] for i in mapped_result]
            mapped_char_result = [phone_char_map[i] for i in phone39_result]
            result_str = remove_duplicate(mapped_char_result)
            f.write('%s,%s\n' % (name, result_str))

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description='Train and evaluate the LSTM model.')
    #ap.add_argument('data_folder', type=str, help='Folder containing train_data.pickle, train_labels.pickle, test_data.pickle and test_labels.pickle.')
    ap.add_argument('-e', '--epochs', type=int, help='Number of epochs for training')
    ap.add_argument('-b', '--batch_size', type=int, help='Size of each minibatch')
    ap.add_argument('-m', '--model', type=str, help='.ckpt file of the model. If -t option is used, evaluate this model. Otherwise, train it.')
    ap.add_argument('-n', '--num_hidden_layers', type=int, help='Number of hidden LSTM layers')
    ap.add_argument('-t', '--test', action='store_true', help='Evaluate a given model.')
    ap.add_argument('-d', '--datadir', type=str, help='data dir')
    ap.add_argument('-o', '--output', type=str, help='output filename')


    ap.set_defaults(epochs = num_epochs,
                    batch_size = train_batch_size, 
                    test = False,
                    size_hidden = num_hidden,
                    num_hidden_layers = num_lstm_layers,
                    optimizer = optimizer_name)
                    
    args = ap.parse_args()

    assert (args.optimizer == 'Adam' or args.optimizer == 'RmsProp'), 'Optimizer must be either "Adam" or "RmsProp": %s' % args.optimizer
    
    optimizer_name = args.optimizer
    num_epochs = args.epochs
    train_batch_size = args.batch_size    
    num_hidden = args.size_hidden
    num_lstm_layers = args.num_hidden_layers


    data_folder = args.datadir
    csv_file_name = args.output

    #data_folder = data_folder + 'fbank'
    #data_folder = data_folder + 'mfcc'
    
    if args.test:

        assert args.model, "Model file is required for evaluation."
        evaluate_rnn_model_from_file(data_folder, args.model)

    else:
        train_rnn(data_folder, args.model)




