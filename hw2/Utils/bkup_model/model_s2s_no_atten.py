import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
from keras.preprocessing import sequence
from tensorflow.python.layers.core import Dense
import random
import json
import argparse

video_train_feat_path = 'training_data/feat'
video_test_feat_path = 'testing_data/feat'

video_train_data_path = './Utils/train_label.csv'
video_test_data_path = './Utils/test_label.csv'

model_save_dir= './model_s2s_no_atten'
outfile = './result.csv'
datadir = 'MLDS_hw2_data'

dim_image = 4096
dim_hidden = 256

video_lstm_step = 80
caption_lstm_step = 20
learning_rate = 0.001
max_gradient_norm = 5

epochs = 205
batch_size = 50
learning_rate = 0.001

word_threshold = 2

def build_model(n_words, bias_init_vector=None):

    video_feat = tf.placeholder(tf.float32, [ batch_size,  video_lstm_step,  dim_image])
    caption = tf.placeholder(tf.int32, [ batch_size,  caption_lstm_step + 2 ])
    
    caption_mask = tf.placeholder(tf.float32, [ batch_size,  caption_lstm_step+1])
    dropout_prob = tf.placeholder(tf.float32,name='Dropout_Keep_Probability')

    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell( dim_hidden) # b t h
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, video_feat, dtype=tf.float32, time_major=False)

    # attention
    #attention_mechanism = tf.contrib.seq2seq.LuongAttention( dim_hidden, encoder_outputs)

    with tf.variable_scope('embedding'):
        embedding_decoder = tf.Variable(tf.truncated_normal(shape=[ n_words,  dim_hidden], stddev=0.1), name='embedding_decoder')
        decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, caption[:,:-1])

    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell( dim_hidden)
    #decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size= dim_hidden)

    decoder_seq_length = [ caption_lstm_step+1] *  batch_size
    # time scheduling
    helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(decoder_emb_inp, decoder_seq_length, embedding_decoder, 0.2, time_major=False)
    #helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_seq_length, time_major=False)
    projection_layer = Dense( n_words, use_bias=False )
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, 
    encoder_state, output_layer = projection_layer)

    # Dynamic decoding
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
    
    logits = outputs.rnn_output
    result = outputs.sample_id

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=caption[:,1:], logits=logits)
    train_loss = (tf.reduce_sum(cross_entropy * caption_mask) / batch_size)

    return train_loss, video_feat, caption, caption_mask, outputs.sample_id, dropout_prob

def build_generator(n_words, bias_init_vector=None):

    batch_size = 1

    video_feat = tf.placeholder(tf.float32, [ batch_size,  video_lstm_step,  dim_image])
    
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell( dim_hidden) # b t h
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, video_feat, dtype=tf.float32, time_major=False)
    # attention
    #attention_mechanism = tf.contrib.seq2seq.LuongAttention( dim_hidden, encoder_outputs)

    with tf.variable_scope('embedding'):
        embedding_decoder = tf.Variable(tf.truncated_normal(shape=[ n_words,  dim_hidden], stddev=0.1), name='embedding_decoder')

    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell( dim_hidden)
    #decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size= dim_hidden)

    # time scheduling
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder, tf.fill([batch_size], 1), 2)

    projection_layer = Dense( n_words, use_bias=False )
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer = projection_layer)

    # Dynamic decoding
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,  maximum_iterations=caption_lstm_step)
    
    result = outputs.sample_id

    generated_words = outputs.sample_id

    return video_feat, generated_words

def preProBuildWordVocab(labels, word_count_threshold=5):
    # borrowed this function from NeuralTalk
    print('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))
    vocabs_count = {}
    nsents = 0

    for label in labels:
        captions = remove_redundent(label['caption'])
        for cap in captions:
            nsents += 1
            for word in cap.lower().split(' '):
                if(word not in vocabs_count): vocabs_count[word] = 0
                vocabs_count[word] += 1

    # ensure word index is same all the times
    vocab = []
    for w in sorted(vocabs_count.keys()):
        if(vocabs_count[w] >= word_count_threshold):
            vocab.append(w)
    print('filtered words from %d to %d' % (len(vocabs_count), len(vocab)))

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

    vocabs_count['<pad>'] = nsents
    vocabs_count['<bos>'] = nsents
    vocabs_count['<eos>'] = nsents
    vocabs_count['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * vocabs_count[ ixtoword[i] ] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector


def remove_redundent(captions):

    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    captions = map(lambda x: x.replace('"', ''), captions)
    captions = map(lambda x: x.replace('\n', ''), captions)
    captions = map(lambda x: x.replace('?', ''), captions)
    captions = map(lambda x: x.replace('!', ''), captions)
    captions = map(lambda x: x.replace('\\', ''), captions)
    captions = map(lambda x: x.replace('/', ''), captions)

    return captions

def train(model_path):

    with open(os.path.join(datadir, 'training_label.json')) as f:
        train_labels = json.load(f)
    with open(os.path.join(datadir, 'testing_label.json')) as f:
        test_labels = json.load(f)

    total_labels = train_labels + test_labels
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(total_labels, word_count_threshold= word_threshold)

    train_data = []
    for data in train_labels:
        videoId = '%s.npy' % (data['id'])
        tmp_data = np.load(os.path.join(datadir, 'training_data', 'feat', videoId))
        train_data.append(tmp_data)
    train_data = np.array(train_data)

    np.save("./Utils/no_atten_wordtoix", wordtoix)
    np.save('./Utils/no_atten_ixtoword', ixtoword)
    np.save("./Utils/no_atten_bias_init_vector", bias_init_vector)

    train_loss, tf_video, tf_caption, tf_caption_mask, tf_probs, do_prob = build_model(len(wordtoix), bias_init_vector=bias_init_vector)

    params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    
    # Optimization
    optimizer = tf.train.AdamOptimizer( learning_rate)
    train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    tf.global_variables_initializer().run()

    #load model if exists
    if model_path is not None:
        saver.restore(sess, model_path)

    train_data = []
    
    for data in train_labels:

        videoId = '%s.npy' % (data['id'])
        tmp_data = np.load(os.path.join(datadir, 'training_data', 'feat', videoId))
        train_data.append(tmp_data)

    train_data = np.array(train_data)

    for epoch in range(0, epochs + 1):

        for start, end in zip(
                range(0, len(train_data), batch_size),
                range(batch_size, len(train_data), batch_size)):

            start_time = time.time()

            current_feats = train_data[start:end]
            current_video_masks = np.zeros((batch_size, video_lstm_step))
            current_captions = []

            for ind in range(len(current_feats)):
                current_video_masks[ind][:len(current_feats[ind])] = 1
                current_captions.append(random.choice(train_labels[start + ind]['caption']))
                
            current_captions = list(map(lambda x: '<bos> ' + x, current_captions))
            current_captions = remove_redundent(current_captions)
            current_captions = list(current_captions)

            current_captions_src = []

            for idx, each_cap in enumerate(current_captions):

                word = each_cap.lower().split(' ')

                if len(word) < caption_lstm_step + 1:
                    current_captions[idx] = current_captions[idx] + ' <eos>'

                else:
                    new_word = ''
                    for i in range(caption_lstm_step):
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


            current_caption_matrix_src = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=caption_lstm_step+1)
            current_caption_matrix_src = np.hstack( [current_caption_matrix_src, np.zeros( [len(current_caption_matrix_src), 1] ) ] ).astype(int)
            
            current_caption_masks = np.zeros( (current_caption_matrix_src.shape[0], current_caption_matrix_src.shape[1]-1) )
            

            nonzeros = np.array( list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix_src )) )

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            update_step, loss_val= sess.run([train_op, train_loss], feed_dict={
                tf_video: current_feats,
                tf_caption: current_caption_matrix_src,
                tf_caption_mask: current_caption_masks
                })

            print('idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))
            
        # loss_val, translate = sess.run([train_loss, tf_probs], feed_dict={
        #         tf_video: current_feats,
        #         tf_caption: current_caption_matrix_src,
        #         tf_caption_mask: current_caption_masks
        #         })       
        
        # generated_word_index = translate[0]
        # generated_words = []

        # for word_idx in generated_word_index:
        #     generated_words.append(ixtoword[word_idx])


        # print(generated_words)
        # print(current_captions)
        
        if np.mod(epoch, 10) == 0:
            print("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, os.path.join(model_save_dir, 'model'), global_step=epoch)

def test(model_path='./models/model-200'):

    test_videos = []
    
    with open(os.path.join(datadir, 'testing_id.txt')) as f:
        for line in f:
            test_videos.append(line.strip())

    ixtoword = pd.Series(np.load('./Utils/no_atten_ixtoword.npy').tolist())
    bias_init_vector = np.load('./Utils/no_atten_bias_init_vector.npy')

    video, words = build_generator(len(ixtoword))
    print(len(ixtoword))
    
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    test_output_txt_fd = open(outfile, 'w')

    for idx, video_feat_path in enumerate(test_videos):

        video_feat = np.load(os.path.join(video_test_feat_path,video_feat_path + '.npy'))[None,...]

        feed_dict={
            video: video_feat
        }

        probs_val = sess.run(words, feed_dict=feed_dict)

        generated_words = ixtoword[list(probs_val[0])]

        punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        video_name = video_feat_path
        
        print(generated_sentence)
        print(idx, video_name)

        test_output_txt_fd.write("%s,%s\n" % (video_name, generated_sentence))

def main():

    ap = argparse.ArgumentParser(description='Train and evaluate the LSTM model.')
    ap.add_argument('-e', '--epochs', type=int, help='Number of epochs for training')
    ap.add_argument('-b', '--batch_size', type=int, help='Size of each minibatch')
    ap.add_argument('-m', '--model', type=str, help='.ckpt file of the model. If -t option is used, evaluate this model. Otherwise, train it.')
    ap.add_argument('-n', '--num_hidden_layers', type=int, help='Number of hidden LSTM layers')
    ap.add_argument('-t', '--test', action='store_true', help='Evaluate a given model.')
    ap.add_argument('-d', '--datadir', type=str, help='data dir')
    ap.add_argument('-o', '--output', type=str, help='output filename')
    ap.add_argument('-s', '--savedir', type=str, help='model save dir')

    global video_train_feat_path
    global video_test_feat_path
    global outfile 
    global epochs

    ap.set_defaults(epochs = epochs,
                batch_size = batch_size, 
                test = False,
                size_hidden = dim_hidden,
                model = None,
                savedir = model_save_dir)

    args = ap.parse_args()

    epochs = args.epochs

    try:    
        os.stat(model_save_dir)
    except:
        os.mkdir(model_save_dir)       

    if not args.datadir:
        print("please provide datadir -d")
        exit()

    if args.output:
        outfile = args.output

    video_train_feat_path = os.path.join(args.datadir, 'training_data', 'feat')
    video_test_feat_path = os.path.join(args.datadir, 'testing_data', 'feat')

    if args.test:
        assert args.model, "Model file is required for evaluation."
        test(model_path=args.model)
    else:
        train(model_path=args.model)
    
if __name__ == "__main__":
    main()
