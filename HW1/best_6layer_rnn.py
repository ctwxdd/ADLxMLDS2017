import tensorflow as tf
import numpy as np
import pickle
import time
import argparse
import os 
import sys
sys.path.append("./Utils")
from Utils import import_data
from Utils import layers
from Utils import map_reader
from tensorflow.contrib.rnn import BasicLSTMCell

data_folder = './data/mfcc_data'
NUM_CLASSES = 48
model_name = 'model_rnn_3'
### Parameters (overidden by argparse, default values listed here)
num_epochs = 15
train_batch_size = 64
num_hidden = 64
num_lstm_layers = 3
use_dropout = True
optimizer_name='Adam'
learn_rate = 0.0001
contex = 1
keep_p = 0.75

### Internal variables
num_features = 39
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

    return tf.reduce_mean(cross_entropy)

def mask_accuracy(output, target):

    mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
    #mask = tf.Print(mask, [mask], summarize=500 )
    correct_prediction = tf.cast(tf.equal(tf.argmax(output, 2), tf.argmax(target, 2)), tf.float32)
    correct_prediction *= mask
    correct_prediction = tf.reduce_sum(correct_prediction, 1)
    correct_prediction /= tf.reduce_sum(mask, 1)

    return tf.reduce_mean(correct_prediction)


def BiRNN(X):

    X = tf.contrib.layers.batch_norm(X)


    with tf.name_scope("conv_1"):
        X = tf.reshape(X, [train_batch_size, -1, num_features, 1])
        W_conv1 = weight_variable([3, 1, 1, 16])
        b_conv1 = bias_variable([16])
        X = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
        #X = tf.Print(X, [tf.shape(X)])
        
    with tf.name_scope("inlayer"):

        X = tf.reshape(X, [-1, num_features])
        W_inlayer = weight_variable([num_features, num_hidden])
        b_inlayer = bias_variable([num_hidden])

        X_in = tf.matmul(X, W_inlayer) + b_inlayer
        X_in = tf.reshape(X_in, [train_batch_size, -1, num_hidden])
        #lstm modules

        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        fw_state = lstm_fw_cell.zero_state(train_batch_size, dtype=tf.float32)
        bw_state = lstm_bw_cell.zero_state(train_batch_size, dtype=tf.float32)
        

        raw_rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, X_in,
                                                sequence_length=sequence_len, initial_state_fw=fw_state, initial_state_bw=bw_state, dtype=tf.float32)
        rnn_out= tf.concat(raw_rnn_out, 2)
        rnn_out = tf.reshape(rnn_out,[-1, num_hidden * 2])

        #output layer
        W_outlayer = weight_variable([num_hidden * 2, num_classes])
        b_outlayer = bias_variable([num_classes])
        results = tf.matmul(rnn_out, W_outlayer) + b_outlayer

        results = tf.reshape(results, [train_batch_size, -1, num_classes])
    
    print("Model creation done")
    
    return results

def BiRNN2(X):

    X = tf.contrib.layers.batch_norm(X)

    # with tf.name_scope("conv_1"):
    #     X = tf.reshape(X, [train_batch_size, -1, num_features, 1])
    #     W_conv1 = weight_variable([3, 1, 1, 16])
    #     b_conv1 = bias_variable([16])
    #     X = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
    #     #X = tf.Print(X, [tf.shape(X)])
        

    with tf.name_scope("inlayer"):

        X = tf.reshape(X, [-1, num_features])
        W_inlayer = weight_variable([num_features, num_hidden])
        b_inlayer = bias_variable([num_hidden])

        X_in = tf.matmul(X, W_inlayer) + b_inlayer
        X_in = tf.reshape(X_in, [train_batch_size, -1, num_hidden])
        #lstm modules
        fw_cell = []
        bw_cell = []
        
        for n in range(num_lstm_layers):

            fw_cell.append(tf.contrib.rnn.BasicLSTMCell(num_hidden))    
            bw_cell.append(tf.contrib.rnn.BasicLSTMCell(num_hidden))

        lstm_fw_cell = tf.contrib.rnn.MultiRNNCell(fw_cell, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.MultiRNNCell(bw_cell, state_is_tuple=True)

        fw_state = lstm_fw_cell.zero_state(train_batch_size, dtype=tf.float32)
        bw_state = lstm_bw_cell.zero_state(train_batch_size, dtype=tf.float32)
        
        #lstm_fw_celll = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, input_keep_prob=keep_prob)
        #lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, input_keep_prob=keep_prob)

        raw_rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, X_in,
                                                sequence_length=sequence_len, initial_state_fw=fw_state, initial_state_bw=bw_state, dtype=tf.float32)
        fw_rnn_out, bw_rnn_out = raw_rnn_out
        rnn_out= tf.concat([fw_rnn_out, bw_rnn_out], 2)
        rnn_out = tf.reshape(rnn_out,[-1, num_hidden * 2])
        #output layer
        W_outlayer = weight_variable([num_hidden * 2, num_classes])
        b_outlayer = bias_variable([num_classes])
        results = tf.matmul(rnn_out, W_outlayer) + b_outlayer

        results = tf.reshape(results, [train_batch_size, -1, num_classes])
    
    print("Model creation done")
    
    return results


def train(train_set, eval_set, y, cost, optimizer):

    with tf.name_scope('accuracy'):
        accuracy = mask_accuracy(y, y_)
    
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
                keep_prob: keep_p
            }                        
            
            if batch%50 == 0:
                train_accuracy = sess.run([ accuracy],feed_dict=feed_dict)  
                print("epoch %d, batch %d, training accuracy %g"%(epoch, batch, train_accuracy[0]))
            
            c_e, _ = sess.run([cost, optimizer], feed_dict=feed_dict)

            if batch%50 == 0:
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

        while not end_of_cross_val:

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
    #y = RNN_CNN(x)
    y = BiRNN(x)
    print("Loading training pickles..")   

    train_set, eval_set = import_data.load_dataset(data_folder + '/train_data.pkl', 
                                         data_folder + '/train_mapped_label.pkl',
                                         batch_size=train_batch_size)
    print("Loading done")
    
    global sess
    global train_writer
    global saver
    saver = tf.train.Saver()

    # Create the dir for the model
    if not os.path.isdir('%s/%s/%s'%(save_loc, model_name, start_date)):
        try:
            os.makedirs('%s/%s/%s'%(save_loc, model_name, start_date))
        except OSError:
            if not os.path.isdir('%s/%s/%s'%(save_loc, model_name, start_date)):
                raise

    cost = maskcost(y, y_)
    #optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate).minimize(cost)
    optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
    
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
    y = BiRNN(x)

    saver = tf.train.Saver()
    global sess

    with tf.Session() as sess:
        saver.restore(sess, model_file)
        print("Model restored")
        evaluate_rnn(data_folder, y)

def evaluate_rnn(data_folder, y):

    r = tf.argmax(y,2)
    test_set = import_data.load_test_dataset(data_folder + '/test_data.pkl', data_folder + '/test_name.pkl', batch_size=1)

    phones_pred = []
    phones_true = []
    frame_accuracies = []
    cross_entropies = []

    end_of_test = False

    phone_list = map_reader.phone_list('./data/48phone_char.map')
    phone_char_map = map_reader.phone_char_reader('./data/48phone_char.map')
    phone_map = map_reader.phone_map_reader('./data/phones/48_39.map')
    
    f = open('out.csv', 'w')
    f.write('id,phone_sequence\n')

    while not end_of_test:

        batch_data, seq_len, name_list = test_set.next_test_batch(1, _pad=contex)

        if batch_data is None:
            end_of_test = True
            break

        feed_dict = {
            x: batch_data,
            sequence_len: seq_len,
            keep_prob: 1.0
        }
        
        result = sess.run([r], feed_dict=feed_dict)[0]
        mapped_result = [phone_list[i] for i in result[0]]

        # for i in range(1,len(mapped_result)-1):
        #     if (mapped_result[i] != mapped_result[i-1] and mapped_result[i-1] == mapped_result[i+1]):
        #         print(mapped_result[i-10 : i+10])
        #         print('%s to %s' % (mapped_result[i] , mapped_result[i-1]))
        #         mapped_result[i] = mapped_result[i-1]
        #         continue

        #     if (mapped_result[i+1] != mapped_result[i-1] and mapped_result[i] != mapped_result[i+1] and mapped_result[i] != mapped_result[i-1]):
        #         print(mapped_result[i-10 : i+10])
        #         print('%s tooo %s' % (mapped_result[i] , mapped_result[i-1]))
        #         mapped_result[i] = mapped_result[i-1]

        phone39_result = [phone_map[i] for i in mapped_result]
        mapped_char_result = [phone_char_map[i] for i in phone39_result]
        result_str = remove_duplicate(mapped_char_result)
        #print('%s,%s' % (name_list[0], result_str))
        f.write('%s,%s\n' % (name_list[0], result_str))


def remove_duplicate(raw_list):
    rst = []
    current = 'L'
    for i in raw_list:
        if i != current:
            rst.append(i)
            current = i
    result = ''.join(rst).strip('L')
    return result
            
        


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Train and evaluate the LSTM model.')
    #ap.add_argument('data_folder', type=str, help='Folder containing train_data.pickle, train_labels.pickle, test_data.pickle and test_labels.pickle.')
    ap.add_argument('-e', '--epochs', type=int, help='Number of epochs for training')
    ap.add_argument('-b', '--batch_size', type=int, help='Size of each minibatch')
    ap.add_argument('-m', '--model', type=str, help='.ckpt file of the model. If -t option is used, evaluate this model. Otherwise, train it.')
    ap.add_argument('-n', '--num_hidden_layers', type=int, help='Number of hidden LSTM layers')
    ap.add_argument('-o', '--optimizer', type=str, help='Optimizer. Either "Adam" or "GradDesc"')
    ap.add_argument('-t', '--test', action='store_true', help='Evaluate a given model.')

    ap.set_defaults(epochs = num_epochs,
                    batch_size = train_batch_size, 
                    test = False,
                    size_hidden = num_hidden,
                    num_hidden_layers = num_lstm_layers,
                    optimizer = optimizer_name)
                    
    args = ap.parse_args()

    assert (args.optimizer == 'Adam' or args.optimizer == 'GradDesc'), 'Optimizer must be either "Adam" or "GradDesc": %s' % args.optimizer
    optimizer_name = args.optimizer

    num_epochs = args.epochs
    train_batch_size = args.batch_size    
    num_hidden = args.size_hidden
    num_lstm_layers = args.num_hidden_layers
    
    if args.test:
        assert args.model, "Model file is required for evaluation."
        evaluate_rnn_model_from_file(data_folder, args.model)
    else:
        train_rnn(data_folder, args.model)


