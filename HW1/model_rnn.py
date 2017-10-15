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
from tensorflow.contrib.rnn import BasicLSTMCell

data_folder = './data'

NUM_CLASSES = 48

### Parameters (overidden by argparse, default values listed here)
num_epochs = 15
train_batch_size = 32
num_hidden = 650
num_lstm_layers = 2
num_steps = 800
use_dropout = True
optimizer_name='Adam'

### Internal variables
num_features = 39
num_classes = 48
start_date = time.strftime("%d-%m-%y/%H.%M.%S")
save_loc = "./output"

x = tf.placeholder(tf.float32, [None, None, num_features], name='input')

# Times when we want to have an early stop (length = batch_size)
batch_size = tf.placeholder(tf.int32, name='batch_size')

# Output classes (true values).
y_ = tf.placeholder(tf.float32, [None, None, num_classes], name='y_')

true_labels = tf.reshape(y_, [-1, num_classes])

# Sequence length for dynamic_rnn()
sequence_len = tf.placeholder(tf.float32, [None], name='sequence_len')

# Probability of keeping nodes at dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

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

def RNN(X):
    """Create Grapth"""
    with tf.name_scope("inlayer"):
        X = tf.reshape(X, [-1, num_features])
        X_in = tf.matmul(X, weights['in']) + b['in']
        X_in = tf.reshape(X_in, [train_batch_size, -1, num_hidden])

    with tf.name_scope("RNN_CELL"):
        lstm_cell = BasicLSTMCell(num_hidden)
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
        #correct_prediction = tf.Print(cp, [tf.argmax(y, 1), tf.argmax(true_labels, 1)], message="y, y_")
        
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

    global summary_op
    global train_writer
    
    global_step = 0
    
    for epoch in range(num_epochs):
        print("epoch %d:" % epoch)
        epoch_done = False
        batch = 0
        while not epoch_done:
            global_step += 1
            batch += 1
            batch_data, batch_labels, seq_len = train_set.next_batch(train_batch_size)
            if batch_data is None:
                # Epoch is finished
                epoch_done = True
                break

            feed_dict = {
                x: batch_data,
                y_: batch_labels,
                sequence_len: seq_len,
                batch_size: train_batch_size,
                keep_prob: 1.0
            }                        
            
            if batch%50 == 0:
                train_accuracy = sess.run([accuracy],feed_dict=feed_dict)           
                print("epoch %d, batch %d, training accuracy %g"%(epoch, batch, train_accuracy[0]))

            feed_dict[keep_prob] = 0.8
            
            c_e, _ = sess.run([cost, optimizer], feed_dict=feed_dict)

            if batch%50 == 0:
                print("cross entropy: %g"%c_e)
            
        train_set.reset_epoch(train_batch_size)
        

        
        end_of_eval = False
        end_of_vali = False
        accuracies_val = []
        accuracies = []

        while not end_of_eval:

            batch_data, batch_labels, seq_len = train_set.next_batch(train_batch_size)

            if batch_data is None:
                end_of_eval = True
                break

            feed_dict = {
                x: batch_data,
                y_: batch_labels,
                sequence_len: seq_len,
                batch_size: train_batch_size,
                keep_prob: 1.0
            }
            
            acc = sess.run([accuracy], feed_dict=feed_dict)
            accuracies.append(acc)

        train_set.reset_epoch(train_batch_size)

        while not end_of_vali:

            batch_data, batch_labels, seq_len = eval_set.next_batch(train_batch_size)

            if batch_data is None:
                end_of_vali = True
                break

            feed_dict = {
                x: batch_data,
                y_: batch_labels,
                sequence_len: seq_len,
                batch_size: train_batch_size,
                keep_prob: 1.0
            }
            
            acc = sess.run([accuracy], feed_dict=feed_dict)
            accuracies_val.append(acc)

        eval_set.reset_epoch(train_batch_size)

        acc_mean = np.mean(np.array(accuracies))
        eval_acc_mean = np.mean(np.array(accuracies_val))
        print("epoch %d finished, accuracy: %g" % (epoch, acc_mean))
        print("valication accuracy: %g" % (eval_acc_mean))

        save_path = saver.save(sess, "%s/models/%s/epoch%d_model.ckpt"%(save_loc, start_date, epoch))
        print("Model for epoch %d saved in file: %s"%(epoch, save_path))

    
def train_rnn(data_folder, model_file = None):
    y = RNN(x)
    
    print("Loading training pickles..")   

    train_set, eval_set = import_data.load_dataset(data_folder + '/train_data.pkl', 
                                         data_folder + '/train_mapped_label.pkl',
                                         keep_sentences=True,
                                         context_frames=1,
                                         seq_length=1,
                                         batch_size=train_batch_size)

    print("Loading done")
    
    global sess
    global summary_op
    global train_writer
    global saver
    saver = tf.train.Saver()

    # Create the dir for the model
    if not os.path.isdir('%s/models/%s'%(save_loc,start_date)):
        try:
            os.makedirs('%s/models/%s'%(save_loc,start_date))
        except OSError:
            if not os.path.isdir('%s/models/%s'%(save_loc,start_date)):
                raise
    
    sess = tf.InteractiveSession()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_labels, logits=y))
    # Optimizer
    # For gradient descend, learning rate = 0.002 (see Hinton et al.)
    # For AdamOptimizer, learning rate = 0.001 (default)
    if (optimizer_name == 'Adam'):

        temp = set(tf.all_variables())
        optimizer = tf.train.AdamOptimizer().minimize(cost)
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
    
    save_path = saver.save(sess, "%s/models/%s/model.ckpt"%(save_loc,start_date))
    print("Model saved in file: %s" % save_path)
    print("Summaries written to %s/summaries/%s" % (save_loc, start_date))
    
    #evaluate_rnn(data_folder, y)
    
# def evaluate_rnn_model_from_file(data_folder, model_file):
#     y, rnn_state = RNN(x)
#     # Op for saving and restoring the model
#     saver = tf.train.Saver()
#     global sess
#     sess = tf.InteractiveSession()
#     saver.restore(sess, model_file)
#     print("Model restored")
    
#     evaluate_rnn(data_folder, y, rnn_state)

# def evaluate_sentence(test_set, y, rnn_state, times, i):
#     end_of_sentence = False
    
#     prediction = []
#     true_labels = []    
    
#     last_state = np.zeros([1, num_hidden*2*num_lstm_layers])
    
#     while not end_of_sentence:
#         data, labels, seq_len = test_set.next_batch(train_batch_size)
#         if data is None:
#             # Entire test set is done
#             return None, None, None, None
        
#         feed_dict = {
#             x: data,
#             y_: labels,
#             batch_size: 1,
#             sequence_len: seq_len,
#             keep_prob: 1.0
#         }                        
        
#         pred, last_state = sess.run([y, rnn_state], feed_dict=feed_dict)
#         prediction.append(pred)
#         true_labels.append(labels[0])
    
#     prediction = np.reshape(prediction, [-1, num_classes])
#     true_labels = np.reshape(true_labels, [-1, num_classes])
#     # Compute the frame accuracy
#     correct_prediction = np.equal(evaluation.fold_labels(np.argmax(prediction, 1)), 
#                                   evaluation.fold_labels(np.argmax(true_labels, 1)))
#     frame_acc = np.mean(np.asarray(correct_prediction, float))
    
#     phones_pred, phones_true, cross_entropy = evaluation.frames_to_phones(prediction, true_labels, times)
#     return phones_pred, phones_true, frame_acc, sess.run(cross_entropy)
        
# def evaluate_rnn(data_folder, y, rnn_state):
#     # For evaluation, we run the same loop as in training, 
#     # without optimization. The batch size remains the same, because a higher
#     # batch size would lead to less sentences being evaluated
#     test_set = import_data.load_dataset(data_folder + '/test_data.pickle', 
#                                          data_folder + '/test_labels.pickle',
#                                          keep_sentences=True,
#                                          context_frames=1,
#                                          seq_length=num_steps,
#                                          batch_size=1)
                                         
#     with open(data_folder + '/test_times.pickle', 'rb') as times_dump:
#         times = cPickle.load(times_dump)

#     # Create arrays that will contain evaluation metrics per sentence    
#     phones_pred = []
#     phones_true = []
#     frame_accuracies = []
#     cross_entropies = []
    
#     for i in range(len(times)):
#         print(i)
#         pred, true_l, frame_acc, cross_entropy = evaluate_sentence(test_set, y, rnn_state, times[i],i)    
#         if frame_acc is None:
#             break
#         frame_accuracies.append(frame_acc)
#         phones_pred.append(pred)
#         phones_true.append(true_l)
#         cross_entropies.append(cross_entropy)
    
#     phones_pred = np.hstack(phones_pred)
#     phones_true = np.hstack(phones_true)
    
#     phone_accuracy = np.mean(np.equal(phones_pred, phones_true))
    
#     # Get the total accuraries
#     frame_accuracy = np.mean(frame_accuracies)
#     cross_entropy = np.mean(cross_entropies)
    
#     print("Test frame accuracy: %f"%frame_accuracy)
#     print("Test phone accuracy: %f"%phone_accuracy)
#     print("Average phone-level cross entropy: %f"%cross_entropy)
#     evaluation.save_confusion_matrix(phones_true, phones_pred, start_date)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Train and evaluate the LSTM model.')
    #ap.add_argument('data_folder', type=str, help='Folder containing train_data.pickle, train_labels.pickle, test_data.pickle and test_labels.pickle.')
    ap.add_argument('-e', '--epochs', type=int, help='Number of epochs for training')
    ap.add_argument('-b', '--batch_size', type=int, help='Size of each minibatch')
    ap.add_argument('-d', '--dropout', action='store_false', help="Don't apply dropout regularization")
    ap.add_argument('-l', '--seq_length', type=int, help='Number of frames per subsequence.')
    ap.add_argument('-m', '--model', type=str, help='.ckpt file of the model. If -t option is used, evaluate this model. Otherwise, train it.')
    ap.add_argument('-n', '--num_hidden_layers', type=int, help='Number of hidden LSTM layers')
    ap.add_argument('-o', '--optimizer', type=str, help='Optimizer. Either "Adam" or "GradDesc"')
    ap.add_argument('-s', '--size_hidden', type=int, help='Number of neurons in each hidden LSTM layer')
    ap.add_argument('-t', '--test', action='store_true', help='Evaluate a given model.')
    
    ap.set_defaults(epochs=15,
                    batch_size=32, 
                    test=False,
                    size_hidden=650,
                    num_hidden_layers=1,
                    seq_length=800,
                    dropout=True,
                    optimizer='Adam')
                    
    args = ap.parse_args()

    assert (args.optimizer == 'Adam' or args.optimizer == 'GradDesc'), 'Optimizer must be either "Adam" or "GradDesc": %s'%args.optimizer
    optimizer_name = args.optimizer

    num_epochs = args.epochs
    train_batch_size = args.batch_size    
    num_hidden = args.size_hidden
    num_lstm_layers = args.num_hidden_layers
    num_steps = args.seq_length
    use_dropout = args.dropout
    
    if args.test:
        assert args.model, "Model file is required for evaluation."
        evaluate_rnn_model_from_file(args.data_folder, args.model)
    else:
        train_rnn(data_folder, args.model)


