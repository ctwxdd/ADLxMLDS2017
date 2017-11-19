import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
from keras.preprocessing import sequence
import argparse

import random

class Video_Caption_Generator():

    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):

        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_video_lstm_step=n_video_lstm_step
        self.n_caption_lstm_step=n_caption_lstm_step

        with tf.device("/gpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')
            #self.Wemb = tf.Variable(tf.truncated_normal([n_words, dim_hidden], stddev=6/math.sqrt(dim_hidden)), name='Wemb')

        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')

        self.attention_W = tf.Variable(tf.truncated_normal([dim_hidden * 2, dim_hidden], -0.01, 0.01), name='attention_W')
        self.attention_b = tf.Variable(tf.zeros([dim_hidden]), name='attention_b')

        self.attention_c_W = tf.Variable(tf.truncated_normal([2*dim_hidden, dim_hidden], -0.01, 0.01), name='attention_W')
        self.attention_c_b = tf.Variable(tf.zeros([dim_hidden]), name='attention_b')


        self.attention_cb_W = tf.Variable(tf.truncated_normal([1,1, dim_hidden, dim_hidden], -0.01, 0.01), name='attention_W_cb')
    
    
        if bias_init_vector is not None:
            
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def content_based(self,hidden, decoder_hidden_state, initializer=None):
        #luong general
        # size of decoder layers
        attention_vec_size = hidden.get_shape()[2].value

        with tf.variable_scope("luong_general", initializer=initializer):

            # here we calculate the W_a * s_i-1 (W1 * h_1) part of the attention alignment
            
            hidden_features = tf.nn.conv2d(hidden, self.attention_cb_W, [1, 1, 1, 1], "SAME")
            
            s = tf.reduce_sum((hidden_features * decoder_hidden_state), [2, 3])

        return s


    def global_attention(self, decoder_hidden_state, hidden_attn, initializer, window_size=10,
                     content_function = None, dtype=tf.float32):

        assert content_function is not None

        attention_vec_size = hidden_attn.get_shape()[2].value
        attn_length = hidden_attn.get_shape()[1].value

        with tf.variable_scope("AttentionGlobal", initializer=initializer):

            hidden_attn = tf.reshape(hidden_attn, [-1, self.n_video_lstm_step, 1, dim_hidden])

            s = content_function(hidden_attn, decoder_hidden_state)
            alpha = tf.nn.softmax(s)

            d = tf.reduce_sum(tf.reshape(alpha, [-1, attn_length, 1, 1]) * hidden_attn, [1, 2])

            ds = tf.reshape(d, [-1, attention_vec_size])#

        return ds

    def linear(self, args, output_size, bias, matrix, bias_term, initializer=None, scope=None):

        assert args is not None

        if not isinstance(args, (list, tuple)):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0

        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
            else:
                total_arg_size += shape[1]

        # Now the computation.
        with tf.variable_scope("Linear"):
            #matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
            
            if len(args) == 1:
                res = tf.matmul(args[0], matrix)
            else:
                res = tf.matmul(tf.concat(args,1), matrix)

            if not bias:
                return res
            #bias_term = tf.get_variable("Bias", [output_size],initializer=tf.constant_initializer(bias_start))

        return res + bias_term



    def build_model(self):

        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1])

        with tf.variable_scope("s2s",  reuse=True):

            video_flat = tf.reshape(video, [-1, self.dim_image])
            image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b )
            image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

            state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
            state2 = tf.zeros([self.batch_size, self.lstm2.state_size])

            padding = tf.zeros([self.batch_size, self.dim_hidden])

            probs = []
            loss = 0.0

        for i in range(0, self.n_video_lstm_step):

            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:,i,:], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2) #output b h
                output2 = tf.reshape(output2, [self.batch_size,1, self.dim_hidden])

                if i == 0:
                    attention_states = output2
                else:
                    attention_states = tf.concat([attention_states ,output2], 1) #(b,n,h) 

        #atten = b n h
        
        print(attention_states)
        # attention_X = tf.reshape(attention_states, [-1, self.n_video_lstm_step]) # (b x h) x n
        # attention = tf.nn.xw_plus_b(attention_X, self.attention_W, self.attention_b) # (b x h) x n
        # attention = tf.reshape(attention, [self.batch_size, self.dim_hidden, self.n_video_lstm_step]) # b x h x n
        # attention = tf.reduce_sum(attention, 2) # b x h


        attn_size = attention_states.get_shape()[2].value
        batch_attn_size = tf.stack([self.batch_size, attn_size])

        ct = tf.zeros( batch_attn_size, dtype= tf.float32)
        ct.set_shape([None, attn_size])
        print(attention_states)

        max_prob_index = tf.ones([batch_size], dtype=tf.int32)

        for i in range(0, self.n_caption_lstm_step):

            if i == 0:
                current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([batch_size], dtype=tf.int64))
            
            else:
                cap_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])
                out_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)

                select_sampler = tf.distributions.Bernoulli(probs=0.2, dtype=tf.bool)

                select_sample = select_sampler.sample(sample_shape=batch_size)
                sample_ids = tf.where(select_sample, max_prob_index , tf.fill([batch_size], -1))
                
                if_sample = (sample_ids > -1)
                current_embed = tf.where(if_sample, out_embed, cap_embed)


            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            print(attention_states)
            ct = self.global_attention(output2, attention_states, None, content_function=self.content_based, dtype=tf.float32)

            #output2 = self.linear([ct] + [output2], self.dim_hidden, True, self.attention_W, self.attention_b)
            print(ct)
            output2 = tf.concat([ct, output2], 1)
            output2 = tf.tanh(output2)

            output2 = tf.nn.xw_plus_b(output2, self.attention_c_W, self.attention_c_b)


            labels = tf.expand_dims(caption[:, i+1], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            

            max_prob_index = tf.to_int32(tf.argmax(logit_words, axis=1))

            
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels , logits=logit_words)
            
            
            
            cross_entropy = cross_entropy * caption_mask[:,i]
            probs.append(tf.argmax(logit_words,1))

            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
            loss = loss + (2-i/20)*current_loss
        
        return loss, video, caption, caption_mask, probs

    

    def build_generator(self):

        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])
        
        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b )
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        for i in range(0, self.n_video_lstm_step):

            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:,i,:], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2) #output b h
                
                output2 = tf.reshape(output2, [self.batch_size, 1, self.dim_hidden])

                if i == 0:
                    attention_states = output2
                else:
                    attention_states = tf.concat([attention_states ,output2], 1) 

        #atten = b n h

        attn_size = attention_states.get_shape()[2].value
        batch_attn_size = tf.stack([self.batch_size, attn_size])

        ct = tf.zeros( batch_attn_size, dtype= tf.float32)
        ct.set_shape([None, attn_size])
        print(attention_states)

        for i in range(0, self.n_caption_lstm_step):
            
            if i == 0:
                with tf.device('/gpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            with tf.variable_scope(tf.get_variable_scope(), reuse=False):

                ct = self.global_attention(output2, attention_states, None, content_function=self.content_based, dtype=tf.float32)

                output2 = self.linear([ct] + [output2], self.dim_hidden, True, self.attention_W, self.attention_b)
                output2 = tf.tanh(output2)

            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)

            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/gpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, generated_words, probs, embeds



video_train_feat_path = 'training_data/feat'
video_test_feat_path = 'testing_data/feat'

video_train_data_path = './Utils/train_label.csv'
video_test_data_path = './Utils/test_label.csv'

model_dir= './models_atten_mod'
outfile = './result.csv'

try:
    os.stat(model_dir)
except:
    os.mkdir(model_dir)       

dim_image = 4096
dim_hidden= 512

n_video_lstm_step = 80
n_caption_lstm_step = 20
n_frame_step = 80

n_epochs = 200
batch_size = 100
learning_rate = 0.001

def get_video_train_data(video_data_path, video_feat_path):

    video_data = pd.read_csv(video_data_path, sep='\t', encoding = "ISO-8859-1")
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID'] + '.npy', axis=1)
    
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    unique_filenames = sorted(video_data['video_path'].unique())
    train_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
    
    return train_data

def get_video_test_data(video_data_path, video_feat_path):

    video_data = pd.read_csv(video_data_path, sep='\t', encoding = "ISO-8859-1")

    video_data['video_path'] = video_data.apply(lambda row: row['VideoID'] + '.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    unique_filenames = sorted(video_data['video_path'].unique())
    test_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]

    return test_data 

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5, largest=None):
    
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1

    if largest:
        cnt = sorted(word_counts.values())
        print(type(cnt))
        word_count_threshold = cnt[largest-1]

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
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector

def train(model_path = None):
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

    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=2)
    
    #np.save("./Utils/wordtoix", wordtoix)
    #np.save('./Utils/ixtoword', ixtoword)
    #np.save("./Utils/bias_init_vector", bias_init_vector)
    ixtoword = pd.Series(np.load('./Utils/ixtoword.npy').tolist())
    wordtoix = pd.Series(np.load('./Utils/wordtoix.npy').tolist())
    bias_init_vector = np.load('./Utils/bias_init_vector.npy')

    with tf.variable_scope(tf.get_variable_scope()):

        model = Video_Caption_Generator(
                dim_image=dim_image,
                n_words=len(wordtoix),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_lstm_steps=n_frame_step,
                n_video_lstm_step=n_video_lstm_step,
                n_caption_lstm_step=n_caption_lstm_step,
                bias_init_vector=bias_init_vector)

        tf_loss, tf_video, tf_caption, tf_caption_mask, tf_probs = model.build_model()
        sess = tf.InteractiveSession()

        saver = tf.train.Saver()
    
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    
    tf.global_variables_initializer().run()

    if model_path is not None:
        saver.restore(sess, model_path)


    #load eval data
    current_eval_data = test_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
    current_eval_data = current_eval_data.reset_index(drop=True)[:batch_size]
    current_eval_videos = current_eval_data['video_path'].values

    eval_feats = np.zeros((batch_size, n_video_lstm_step, dim_image))
    
    current_eval_feats_vals = list(map(lambda vid: np.load(vid), current_eval_videos))

    for ind,feat in enumerate(current_eval_feats_vals):
        eval_feats[ind][:len(current_eval_feats_vals[ind])] = feat

    current_captions = current_eval_data['Description'].values
    current_captions = list(map(lambda x: '<bos> ' + x, current_captions))
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
        if len(word) < n_caption_lstm_step:
            current_captions[idx] = current_captions[idx] + ' <eos>'
        else:
            new_word = ''
            for i in range(n_caption_lstm_step-1):
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

    current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step)
    current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
    current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
    nonzeros = np.array( list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix )) )




    #eval_feat
    eval_caption_matrix = current_caption_matrix
    eval_caption_masks = current_caption_masks

    for epoch in range(0, n_epochs):

        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.ix[index]
        
        current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
        current_train_data = current_train_data.reset_index(drop=True)
        #print(current_train_data.iloc[0])
        for start, end in zip(
                range(0, len(current_train_data), batch_size),
                range(batch_size, len(current_train_data), batch_size)):

            start_time = time.time()

            current_batch = current_train_data[start:end]
            current_videos = current_batch['video_path'].values

            current_feats = np.zeros((batch_size, n_video_lstm_step, dim_image))
            current_feats_vals = list(map(lambda vid: np.load(vid), current_videos))

            


            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat

            current_captions = current_batch['Description'].values
            current_captions = list(map(lambda x: '<bos> ' + x, current_captions))
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
                if len(word) < n_caption_lstm_step:
                    current_captions[idx] = current_captions[idx] + ' <eos>'
                else:
                    new_word = ''
                    for i in range(n_caption_lstm_step-1):
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

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
            current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
            nonzeros = np.array( list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix )) )

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            _, loss_val = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_video: current_feats,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })

            print('idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))


        probs_val = sess.run(tf_probs, feed_dict={
        tf_video: eval_feats,
        tf_caption: eval_caption_matrix,
        tf_caption_mask: eval_caption_masks,
        })

        n = random.randint(0,49)
        word = []
        for i in probs_val:
            word.append(ixtoword[i[n]])

        print (word)

        capt = []
        for i in eval_caption_matrix[n]:
            capt.append(ixtoword[i])
        print (capt)


        if np.mod(epoch, 10) == 0:
            print("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, os.path.join(model_dir, 'model'), global_step=epoch)

def test(model_path='./models/model-100'):
   
    test_videos = os.listdir(video_test_feat_path)

    ixtoword = pd.Series(np.load('./Utils/ixtoword.npy').tolist())

    bias_init_vector = np.load('./Utils/bias_init_vector.npy')

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    video_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    test_output_txt_fd = open(outfile, 'w')

    for idx, video_feat_path in enumerate(test_videos):

        if '.npy' not in video_feat_path:
            video_feat_path += '.npy'
        
        video_feat = np.load(os.path.join(video_test_feat_path,video_feat_path))[None,...]

        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat})
        generated_words = ixtoword[generated_word_index]

        punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        
        test_output_txt_fd.write("%s,%s\n" % (video_feat_path[:-4], generated_sentence))


def main():

    ap = argparse.ArgumentParser(description='Train and evaluate the LSTM model.')
    #ap.add_argument('data_folder', type=str, help='Folder containing train_data.pickle, train_labels.pickle, test_data.pickle and test_labels.pickle.')
    ap.add_argument('-e', '--epochs', type=int, help='Number of epochs for training')
    ap.add_argument('-b', '--batch_size', type=int, help='Size of each minibatch')
    ap.add_argument('-m', '--model', type=str, help='.ckpt file of the model. If -t option is used, evaluate this model. Otherwise, train it.')
    ap.add_argument('-n', '--num_hidden_layers', type=int, help='Number of hidden LSTM layers')
    ap.add_argument('-t', '--test', action='store_true', help='Evaluate a given model.')
    ap.add_argument('-d', '--datadir', type=str, help='data dir')
    ap.add_argument('-o', '--output', type=str, help='output filename')

    

    global video_train_feat_path
    global video_test_feat_path
    global outfile 
    global n_epochs
    global model_dir
    global batch_size

    ap.set_defaults(epochs = n_epochs,
                batch_size = batch_size, 
                test = False,
                size_hidden = dim_hidden,
                model = None)

    args = ap.parse_args()

    n_epochs = args.epochs

    if not args.datadir:
        print("please provide datadir -d")
        exit()

    if args.output:
        outfile = args.output

    video_train_feat_path = os.path.join(args.datadir, 'training_data', 'feat')
    video_test_feat_path = os.path.join(args.datadir, 'testing_data', 'feat')

    if args.test:
        batch_size = 1
        assert args.model, "Model file is required for evaluation."
        test(model_path=args.model)

    else:
        
        train(model_path=args.model)
    
if __name__ == "__main__":
    main()

