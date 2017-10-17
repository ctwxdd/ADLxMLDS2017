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

data_folder = './data'
NUM_CLASSES = 48

### Parameters (overidden by argparse, default values listed here)
num_epochs = 15
train_batch_size = 32
num_hidden = 500
num_lstm_layers = 2
num_steps = 800
use_dropout = True
optimizer_name='Adam'

### Internal variables
num_features = 69
num_classes = 48
start_date = time.strftime("%d-%m-%y/%H.%M.%S")
save_loc = "./output"

x = tf.placeholder(tf.float32, [None, None, num_features], name='input')

# Times when we want to have an early stop (length = batch_size)
batch_size = tf.placeholder(tf.int32, name='batch_size')

# Output classes (true values).
y_ = tf.placeholder(tf.float32, [None, None, num_classes], name='y_')

true_labels = tf.reshape(y_, [-1, num_classes])

# Initial state for the LSTM
sequence_len = tf.placeholder(tf.float32, [None], name='sequence_len')

# Probability of keeping nodes at dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

### Ops
summary_op = None
train_writer = None
sess = None
# Op for saving and restoring the model
saver = None

weights = {
    'in': tf.Variable(tf.random_uniform([num_features, num_hidden], -1.0, 1.0), name="in_w"),
    'out': tf.Variable(tf.random_uniform([num_hidden, num_classes], -1.0, 1.0), name="out_w"),
}
b = {
    'in': tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="in_bias"),
    'out': tf.Variable(tf.constant(0.1, shape=[num_classes]), name="out_bias"),
}


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def RNN_CNN(X):
    
    with tf.name_scope("conv_1"):
        X = tf.reshape(X, [train_batch_size, -1, num_features, 1])
        W_conv1 = weight_variable([5, 1, 1, 1])
        b_conv1 = bias_variable([1])
        X = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
        X = tf.reshape(X, [train_batch_size, -1, num_features])
        #X = tf.Print(X, [tf.shape(X)])

    with tf.name_scope("inlayer"):
        X = tf.reshape(X, [-1, num_features])
        X_in = tf.matmul(X, weights['in']) + b['in']
        X_in = tf.reshape(X_in, [train_batch_size, -1, num_hidden])

    with tf.name_scope("RNN_CELL"):

        cells = []

        for n in range(num_lstm_layers):
            cells.append(tf.contrib.rnn.BasicLSTMCell(num_hidden, state_is_tuple=True))

        lstm_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        
        
        #lstm_cell = BasicLSTMCell(num_hidden)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,input_keep_prob=keep_prob)
        rnn_out, states = tf.nn.dynamic_rnn(lstm_cell, X_in, sequence_length=sequence_len, dtype=tf.float32)

    with tf.name_scope("out_layer"):
        rnn_out = tf.reshape(rnn_out,[-1, num_hidden])        
        results = tf.matmul(rnn_out, weights['out']) + b['out']
    
    print("Model creation done")
    
    return results


def train(train_set, eval_set, y, cost, optimizer):

    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(true_labels, 1))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    
    for epoch in range(num_epochs):
        print("epoch %d:" % epoch)
        epoch_done = False
        batch = 0
        while not epoch_done:
            batch += 1
            batch_data, batch_labels, seq_len = train_set.next_batch(train_batch_size, _pad=2)
            if batch_data is None:
                # Epoch is finished
                epoch_done = True
                break
            
            # Reset the LSTM State for the sequences that ended, 
            # otherwise use the previous state
            feed_dict = {
                x: batch_data,
                y_: batch_labels,
                sequence_len: seq_len,
                batch_size: train_batch_size,
                keep_prob: 0.75
            }                        
            
            if batch%50 == 0:
                train_accuracy = sess.run([ accuracy],feed_dict=feed_dict)  
                print("epoch %d, batch %d, training accuracy %g"%(epoch, batch, train_accuracy[0]))
            
            feed_dict[keep_prob] = 0.8
            
            c_e, _ = sess.run([cost, optimizer], feed_dict=feed_dict)

            if batch%50 == 0:
                print("cross entropy: %g"%c_e)

        num_examples = train_set.data.shape[0]
        train_set.reset_epoch(train_batch_size)
        
        end_of_eval = False

        accuracies = []

        while not end_of_eval:

            batch_data, batch_labels, seq_len = train_set.next_batch(train_batch_size, _pad=2)

            if batch_data is None:
                end_of_eval = True
                break

            feed_dict = {
                x: batch_data,
                y_: batch_labels,
                sequence_len: seq_len,
                batch_size: num_examples,
                keep_prob: 1.0
            }
            
            acc = sess.run([accuracy], feed_dict=feed_dict)
            accuracies.append(acc)

        train_set.reset_epoch(train_batch_size)
        end_of_cross_val = False
        accuracies_val = []

        while not end_of_cross_val:

            batch_data, batch_labels, seq_len = eval_set.next_batch(train_batch_size, _pad=2)

            if batch_data is None:
                end_of_cross_val = True
                break
            # Reset the LSTM State for the sequences that ended, 
            # otherwise use the previous state

            feed_dict = {
                x: batch_data,
                y_: batch_labels,
                sequence_len: seq_len,
                batch_size: num_examples,
                keep_prob: 1.0
            }
            
            acc = sess.run([accuracy], feed_dict=feed_dict)
            accuracies_val.append(acc)

        eval_set.reset_epoch(train_batch_size)

        acc_mean = np.mean(np.array(accuracies))
        eval_acc_mean = np.mean(np.array(accuracies_val))
        print("epoch %d finished, accuracy: %g" % (epoch, acc_mean))
        print("valication accuracy: %g" % (eval_acc_mean))
        save_path = saver.save(sess, "%s/models_cnn/%s/epoch%d_model.ckpt"%(save_loc, start_date, epoch))
        print("Model for epoch %d saved in file: %s"%(epoch, save_path))
        train_set.reset_epoch(train_batch_size)

    
def train_rnn(data_folder, model_file = None):
    y = RNN_CNN(x)
    
    print("Loading training pickles..")   

    # We want to keep the sentences in order to train per sentence
    # Sentences are padded to num_steps
    train_set, eval_set = import_data.load_dataset(data_folder + '/train_data.pkl', 
                                         data_folder + '/train_mapped_label.pkl',
                                         batch_size=train_batch_size)

    print("Loading done")
    
    global sess
    global summary_op
    global train_writer
    global saver
    saver = tf.train.Saver()

    # Create the dir for the model
    if not os.path.isdir('%s/models_cnn/%s'%(save_loc,start_date)):
        try:
            os.makedirs('%s/models_cnn/%s'%(save_loc,start_date))
        except OSError:
            if not os.path.isdir('%s/models_cnn/%s'%(save_loc,start_date)):
                raise
    
    sess = tf.InteractiveSession()
    summary_op = tf.summary.merge_all()    
    train_writer = tf.summary.FileWriter('%s/summaries/%s'%(save_loc, start_date), sess.graph)
        
    # Cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_labels, logits=y))
    # Optimizer
    # For gradient descend, learning rate = 0.002 (see Hinton et al.)
    # For AdamOptimizer, learning rate = 0.001 (default)
    if (optimizer_name == 'Adam'):

        temp = set(tf.all_variables())
        optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
    else:
        optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(cost)
    
    if model_file:
        saver.restore(sess, model_file)
        print("Model restored")
    else:
        # Initialization
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
    
    print("Training network. Date: %s" % start_date)
    train(train_set, eval_set, y, cost, optimizer)
    
    save_path = saver.save(sess, "%s/models_cnn/%s/model.ckpt"%(save_loc,start_date))
    print("Model saved in file: %s" % save_path)
    print("Summaries written to %s/summaries/%s" % (save_loc, start_date))
    
    #evaluate_rnn(data_folder, y)
    
def evaluate_rnn_model_from_file(data_folder, model_file):
    #Load the CNN_RNN graph
    y = RNN_CNN(x)
    saver = tf.train.Saver()
    global sess
    sess = tf.InteractiveSession()
    saver.restore(sess, model_file)
    print("Model restored")
    
    evaluate_rnn(data_folder, y)

        
def evaluate_rnn(data_folder, y):
    # For evaluation, we run the same loop as in training, 
    # without optimization. The batch size remains the same, because a higher
    # batch size would lead to less sentences being evaluated

    r = tf.argmax(y,1)
    test_set = import_data.load_test_dataset(data_folder + '/test_data.pkl', data_folder + '/test_name.pkl', batch_size=1)
    # Create arrays that will contain evaluation metrics per sentence    
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

            batch_data, seq_len, name_list = test_set.next_test_batch(train_batch_size, _pad=True)

            if batch_data is None:
                end_of_test = True
                break
            # Reset the LSTM State for the sequences that ended, 
            # otherwise use the previous state
            feed_dict = {
                x: batch_data,
                sequence_len: seq_len,
                batch_size: 1,
                keep_prob: 1.0
            }
            
            result = sess.run([r], feed_dict=feed_dict)

            
            #print(result)
            #print(result[0].shape)
            mapped_result = [phone_list[i] for i in result[0]]
            #print(mapped_result)
            mapped_char_result = [phone_char_map[i] for i in mapped_result]
            #print(mapped_char_result)
            result_str = remove_duplicate(mapped_char_result)
            print('%s,%s' % (name_list[0], result_str))
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
    ap.add_argument('-d', '--dropout', action='store_false', help="Don't apply dropout regularization")
    ap.add_argument('-m', '--model', type=str, help='.ckpt file of the model. If -t option is used, evaluate this model. Otherwise, train it.')
    ap.add_argument('-n', '--num_hidden_layers', type=int, help='Number of hidden LSTM layers')
    ap.add_argument('-o', '--optimizer', type=str, help='Optimizer. Either "Adam" or "GradDesc"')
    ap.add_argument('-s', '--size_hidden', type=int, help='Number of neurons in each hidden LSTM layer')
    ap.add_argument('-t', '--test', action='store_true', help='Evaluate a given model.')

    ap.set_defaults(epochs = num_epochs,
                    batch_size = train_batch_size, 
                    test = False,
                    size_hidden = num_hidden,
                    num_hidden_layers = num_lstm_layers,
                    dropout = use_dropout,
                    optimizer = optimizer_name)
                    
    args = ap.parse_args()

    assert (args.optimizer == 'Adam' or args.optimizer == 'GradDesc'), 'Optimizer must be either "Adam" or "GradDesc": %s'%args.optimizer
    optimizer_name = args.optimizer

    num_epochs = args.epochs
    train_batch_size = args.batch_size    
    num_hidden = args.size_hidden
    num_lstm_layers = args.num_hidden_layers
    use_dropout = args.dropout
    
    if args.test:
        assert args.model, "Model file is required for evaluation."
        evaluate_rnn_model_from_file(data_folder, args.model)
    else:
        train_rnn(data_folder, args.model)


