import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
from keras.preprocessing import sequence
import random


data_path = sys.argv[1]
outfile = sys.argv[2]


video_train_feat_path = os.path.join(data_path, 'training_data', 'feat') 
video_test_feat_path = os.path.join(data_path, 'testing_data', 'feat')

video_train_data_path = './train_label.csv'
video_test_data_path = './test_label.csv'

model_path = './models'

try:
    os.stat(model_path)
except:
    os.mkdir(model_path)

dim_image = 4096
dim_hidden = 256

video_lstm_step = 80
caption_lstm_step = 20

epochs = 205
batch_size = 50
learning_rate = 0.001

def build_model(n_words, bias_init_vector=None):
        
    Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

    video = tf.placeholder(tf.float32,shape=[batch_size, video_lstm_step, dim_image], name='Input_Video')
    caption = tf.placeholder(tf.int32,shape=[batch_size, caption_lstm_step], name='GT_Caption')

    caption_mask = tf.placeholder(tf.int32,shape=[batch_size, caption_lstm_step],name='Caption_Mask')
    dropout_prob = tf.placeholder(tf.float32,name='Dropout_Keep_Probability')

    encode_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
    encode_image_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_image_b')

    decode_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')

    if bias_init_vector is not None:

        decode_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
    else:

        decode_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    video_emb = tf.reshape(video, [-1, dim_image])
    video_emb = tf.nn.xw_plus_b(video_emb, encode_image_W, encode_image_b)
    
    video_emb = tf.reshape(video_emb, [batch_size, -1, dim_hidden])

    with tf.variable_scope('LSTM_Video',reuse=None) as scope:

        cell_vid = tf.nn.rnn_cell.LSTMCell(dim_hidden)
        #lstm_vid = tf.nn.rnn_cell.DropoutWrapper(lstm_vid,output_keep_prob=dropout_prob)
        ini_state = cell_vid.zero_state(batch_size, tf.float32)
        
        out_vid, state_vid = tf.nn.dynamic_rnn(cell_vid, video_emb, initial_state=ini_state, dtype=tf.float32)

    #Build the decoder
    sos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='SOS')
    pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

    sos_step_embedded = tf.nn.embedding_lookup(Wemb, sos_time_slice)
    pad_step_embedded = tf.nn.embedding_lookup(Wemb, pad_time_slice)

    decoder_length = tf.constant(caption_lstm_step + 1, shape=[batch_size]) 
    encoder_length = tf.constant(video_lstm_step, shape=[batch_size]) 
      
    def initial_fn():

        initial_elements_finished = (0 >= decoder_length)  # all False at the initial step
        initial_input = sos_step_embedded
        
        return initial_elements_finished, initial_input

    def sample_fn(time, outputs, state):

        prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
        return prediction_id

    def next_inputs_fn(time, outputs, state, sample_ids):

        current_embed = tf.nn.embedding_lookup(Wemb, caption[:, time])
        next_input = current_embed
        elements_finished = (time > decoder_length)
        next_state = state

        return elements_finished, next_input, next_state

    helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

    with tf.variable_scope(scope):

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units= dim_hidden, memory=out_vid, memory_sequence_length = encoder_length)
    
        cell = tf.contrib.rnn.LSTMCell(num_units=dim_hidden)

        attn_cell = tf.contrib.seq2seq.AttentionWrapper( cell, attention_mechanism, attention_layer_size=dim_hidden)
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, dim_hidden)

        decoder = tf.contrib.seq2seq.BasicDecoder(cell=out_cell, helper=helper, initial_state=out_cell.zero_state(
                        dtype=tf.float32, batch_size= batch_size))
    
        
        outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,maximum_iterations=caption_lstm_step)

    outputs = tf.reshape(outputs.rnn_output, [-1, dim_hidden])
    logits = tf.nn.xw_plus_b( outputs, decode_word_W, decode_word_b)
    
    logits = tf.reshape(logits, [batch_size, -1, n_words])

    #masking the loss
    mask = tf.to_float(caption_mask)
    # crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= caption, logits= logits)
    # 
    onehot_labels = tf.one_hot(caption, n_words)
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels , logits=logits)
    loss = (tf.reduce_sum(cross_entropy * mask)/batch_size)

    prob = tf.arg_max(logits, 2)

    return loss, video, caption, caption_mask, prob, dropout_prob

def build_generator(n_words, bias_init_vector=None):
        
    Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

    video = tf.placeholder(tf.float32,shape=[batch_size, video_lstm_step, dim_image], name='Input_Video')
    caption = tf.placeholder(tf.int32,shape=[batch_size, caption_lstm_step], name='GT_Caption')

    caption_mask = tf.placeholder(tf.int32,shape=[batch_size, caption_lstm_step],name='Caption_Mask')
    dropout_prob = tf.placeholder(tf.float32,name='Dropout_Keep_Probability')

    encode_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
    encode_image_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_image_b')

    decode_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')

    if bias_init_vector is not None:

        decode_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
    else:

        decode_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    video_emb = tf.reshape(video, [-1, dim_image])
    video_emb = tf.nn.xw_plus_b(video_emb, encode_image_W, encode_image_b)
    
    video_emb = tf.reshape(video_emb, [batch_size, -1, dim_hidden])

    with tf.variable_scope('LSTM_Video',reuse=None) as scope:

        cell_vid = tf.nn.rnn_cell.LSTMCell(dim_hidden)
        #lstm_vid = tf.nn.rnn_cell.DropoutWrapper(lstm_vid,output_keep_prob=dropout_prob)
        ini_state = cell_vid.zero_state(batch_size, tf.float32)
        
        out_vid, state_vid = tf.nn.dynamic_rnn(cell_vid, video_emb, initial_state=ini_state, dtype=tf.float32)

    #Build the decoder
    sos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='SOS')
    pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

    sos_step_embedded = tf.nn.embedding_lookup(Wemb, sos_time_slice)
    pad_step_embedded = tf.nn.embedding_lookup(Wemb, pad_time_slice)

    decoder_length = tf.constant(caption_lstm_step + 1, shape=[batch_size]) 
      
    def initial_fn():

        initial_elements_finished = (0 >= decoder_length)  # all False at the initial step
        initial_input = sos_step_embedded
        
        return initial_elements_finished, initial_input

    def sample_fn(time, outputs, state):

        outputs = tf.reshape(outputs, [-1, dim_hidden ])
        logits = tf.nn.xw_plus_b( outputs, decode_word_W, decode_word_b)
        logits = tf.reshape(logits, [batch_size, n_words])

        max_prob_index = tf.to_int32(tf.argmax(logits, axis=1))
        
        return max_prob_index

    def next_inputs_fn(time, outputs, state, sample_ids):

        current_embed = tf.nn.embedding_lookup(Wemb, sample_ids)
        next_input = current_embed
        elements_finished = (time > decoder_length)
        next_state = state

        return elements_finished, next_input, next_state

    helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

    with tf.variable_scope(scope):

        cell = tf.contrib.rnn.LSTMCell(num_units=dim_hidden)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell, helper=helper,initial_state= state_vid)
        
        outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,maximum_iterations=caption_lstm_step)

    outputs = tf.reshape(outputs.rnn_output, [-1, dim_hidden])
    logits = tf.nn.xw_plus_b( outputs, decode_word_W, decode_word_b)
    
    logits = tf.reshape(logits, [batch_size, -1, n_words])

    #masking the loss
    mask = tf.to_float(caption_mask)
    # crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= caption, logits= logits)
    # 

    onehot_labels = tf.one_hot(caption, n_words)
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels , logits=logits)
    loss = (tf.reduce_sum(cross_entropy * mask)/batch_size)

    prob = tf.arg_max(logits, 2)

    return loss, video, caption, caption_mask, prob, dropout_prob



def get_video_train_data(video_data_path, video_feat_path):

    video_data = pd.read_csv(video_data_path, sep='\t', encoding="ISO-8859-1")
    video_data['video_path'] = video_data.apply(
        lambda row: row['VideoID'] + '.npy', axis=1)

    video_data['video_path'] = video_data['video_path'].map(
        lambda x: os.path.join(video_feat_path, x))
    unique_filenames = sorted(video_data['video_path'].unique())
    train_data = video_data[video_data['video_path'].map(
        lambda x: x in unique_filenames)]

    return train_data


def get_video_test_data(video_data_path, video_feat_path):

    video_data = pd.read_csv(video_data_path, sep='\t', encoding="ISO-8859-1")

    video_data['video_path'] = video_data.apply(
        lambda row: row['VideoID'] + '.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(
        lambda x: os.path.join(video_feat_path, x))
    unique_filenames = sorted(video_data['video_path'].unique())
    test_data = video_data[video_data['video_path'].map(
        lambda x: x in unique_filenames)]

    return test_data


def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    # borrowed this function from NeuralTalk
    print('preprocessing word counts and creating vocab based on word count threshold %d' % (
        word_count_threshold))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx + 4
        ixtoword[idx + 4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector)  # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)  # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector


def train(model = None):

    train_data = get_video_train_data(video_train_data_path, video_train_feat_path)
    train_captions = train_data['Description'].values

    test_data = get_video_test_data(video_test_data_path, video_test_feat_path)
    test_captions = test_data['Description'].values

    captions_list = list(train_captions) + list(test_captions)
    captions = np.asarray(captions_list, dtype=np.object)

    captions = list(map(lambda x: x.replace('.', ''), captions))
    captions = list(map(lambda x: x.replace(',', ''), captions))
    captions = list(map(lambda x: x.replace('"', ''), captions))
    captions = list(map(lambda x: x.replace('\n', ''), captions))
    captions = list(map(lambda x: x.replace('?', ''), captions))
    captions = list(map(lambda x: x.replace('!', ''), captions))
    captions = list(map(lambda x: x.replace('\\', ''), captions))
    captions = list(map(lambda x: x.replace('/', ''), captions))

    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=0)

    np.save("./data/wordtoix", wordtoix)
    np.save('./data/ixtoword', ixtoword)
    np.save("./data/bias_init_vector", bias_init_vector)

    with tf.variable_scope(tf.get_variable_scope()):

        #tf_loss, tf_video, tf_caption, tf_caption_mask, tf_probs, do_prob = build_model(len(wordtoix), bias_init_vector=bias_init_vector)
        tf_loss, tf_video, tf_caption, tf_caption_mask, tf_probs, do_prob = build_model(len(wordtoix), bias_init_vector=bias_init_vector)
        sess = tf.InteractiveSession()


    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)

    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('logs', sess.graph)


    # new_saver = tf.train.import_meta_graph('./rgb_models/model-1000.meta')
    # new_saver.restore(sess, tf.train.latest_checkpoint('./models/'))
    if model is not None:
        saver.restore(sess, model)

    for epoch in range(0, epochs):

        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.ix[index]

        current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
        current_train_data = current_train_data.reset_index(drop=True)


        for start, end in zip(range(0, len(current_train_data), batch_size), range(batch_size, len(current_train_data), batch_size)):

            start_time = time.time()

            current_batch = current_train_data[start:end]
            current_videos = current_batch['video_path'].values

            current_feats = np.zeros((batch_size, video_lstm_step, dim_image))
            current_feats_vals = list(map(lambda vid: np.load(vid), current_videos))
            #shape(80, 4096)
            current_video_masks = np.zeros((batch_size, video_lstm_step))

            current_captions = current_batch['Description'].values

            #current_captions = list(map(lambda x: '<bos> ' + x, current_captions))
            current_captions = list(map(lambda x: x.replace('.', ''), current_captions))
            current_captions = list(map(lambda x: x.replace(',', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('"', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('\n', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('?', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('!', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('\\', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('/', ''), current_captions))

            for idx, each_cap in enumerate(current_captions):

                word = each_cap.lower().split(' ')

                if len(word) < caption_lstm_step:
                    current_captions[idx] = current_captions[idx] + ' <eos>'

                else:
                    new_word = ''
                    for i in range(caption_lstm_step - 1):
                        new_word = new_word + word[i] + ' '

                    current_captions[idx] = new_word + '<eos>'
            
            
            current_caption_ind = []

            for cap in current_captions:
                current_word_ind = []
                for word in cap.lower().split(' '):
                    if word in wordtoix:
                        current_word_ind.append(wordtoix[word])
                    else:
                        current_word_ind.append(wordtoix['<unk>'])

                current_caption_ind.append(current_word_ind)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=caption_lstm_step)
            #create caption mask
            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix)))

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            feed_dict={
                    tf_video: current_feats,
                    tf_caption: current_caption_matrix,
                    tf_caption_mask: current_caption_masks,
                    do_prob: 1.0
                }

            _, loss_val = sess.run(
                [train_op, tf_loss],feed_dict=feed_dict
                )

            
            print('idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))
        
        probs_val = sess.run(tf_probs, feed_dict={
            tf_video: current_feats,
            tf_caption: current_caption_matrix,
            tf_caption_mask: current_caption_masks,
            do_prob: 1.0,
            })

        #print(probs_val[0])
        #print(current_caption_matrix[0])
        #print(current_caption_masks[0])
        n = random.randint(0,49)
        word = []
        for i in probs_val[n]:
            word.append(ixtoword[i])
        print (word)

        capt = []
        for i in current_caption_matrix[n]:
            capt.append(ixtoword[i])
        print (capt)

        tf.summary.scalar('loss', loss_val)


        merged = tf.summary.merge_all()
        summary = sess.run([merged],feed_dict=feed_dict)

        writer.add_summary(summary[0], epoch)
        
        if np.mod(epoch, 10) == 0:
            print("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)


def test(model_path='./models/model-200'):
    test_data = get_video_test_data(video_test_data_path, video_test_feat_path)
    test_videos = test_data['video_path'].unique()

    ixtoword = pd.Series(np.load('./data/ixtoword.npy').tolist())

    bias_init_vector = np.load('./data/bias_init_vector.npy')
    batch_size = 1
    
    with tf.variable_scope(tf.get_variable_scope()):
        #tf_loss, tf_video, tf_caption, tf_caption_mask, tf_probs, do_prob = build_model(len(wordtoix), bias_init_vector=bias_init_vector)
        tf_loss, tf_video, tf_caption, tf_caption_mask, tf_probs, do_prob = build_generator(len(ixtoword))
        sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    test_output_txt_fd = open('S2VT_results.txt', 'w')

    for idx, video_feat_path in enumerate(test_videos):

        video_feat = np.load(video_feat_path)[None, ...]

        feed_dict={
            tf_video: video_feat,
            tf_caption: np.zeros((1, caption_lstm_step)),
            tf_caption_mask: np.zeros((1, caption_lstm_step)),
            do_prob: 1.0
        }

        probs_val = sess.run(tf_probs, feed_dict=feed_dict)

        generated_words = ixtoword[list(probs_val[0])]

        punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        video_name = video_feat_path.split('\\')[1][:-4]
        
        print(generated_sentence)
        print(idx, video_name)

        test_output_txt_fd.write("%s,%s\n" % (video_name, generated_sentence))


def main():

    train(model='./models/model-200')
    global batch_size 
    #batch_size = 1
    #test(model_path='./models/model-200')

if __name__ == "__main__":
    main()
