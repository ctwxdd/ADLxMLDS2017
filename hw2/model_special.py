import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
from keras.preprocessing import sequence
import argparse

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

        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')


        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

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
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

        for i in range(0, self.n_caption_lstm_step):

            with tf.device("/gpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])

            tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat( [current_embed, output1], 1), state2)

            labels = tf.expand_dims(caption[:, i+1], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels , logits=logit_words)
            cross_entropy = cross_entropy * caption_mask[:,i]
            probs.append(logit_words)

            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
            loss = loss + current_loss

        return loss, video, caption, caption_mask, probs


    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        for i in range(0, self.n_video_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:, i, :], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

        for i in range(0, self.n_caption_lstm_step):
            tf.get_variable_scope().reuse_variables()

            if i == 0:
                with tf.device('/gpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

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

model_dir= './models_special_2'
outfile = './result.csv'

try:
    os.stat(model_dir)
except:
    os.mkdir(model_dir)       

dim_image = 4096
dim_hidden= 256

n_video_lstm_step = 80
n_caption_lstm_step = 20
n_frame_step = 80

n_epochs = 200
batch_size = 50
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

    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=0)
    
    np.save("./Utils/wordtoix", wordtoix)
    np.save('./Utils/ixtoword', ixtoword)
    np.save("./Utils/bias_init_vector", bias_init_vector)

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

        saver = tf.train.Saver(write_version=1)
    
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    
    tf.global_variables_initializer().run()

    if model_path is not None:
        saver.restore(sess, model_path)



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

            current_video_masks = np.zeros((batch_size, n_video_lstm_step))

            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1

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
                        tf_video_mask : current_video_masks,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })

            print('idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))

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
        
        video_feat = np.load(os.path.join(video_test_feat_path,video_feat_path))[None,...]

        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat})
        generated_words = ixtoword[generated_word_index]

        punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        
        test_output_txt_fd.write("%s,%s\n" % (video_feat_path, generated_sentence))


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

    ap.set_defaults(epochs = n_epochs,
                batch_size = batch_size, 
                test = False,
                size_hidden = dim_hidden,
                model = None)

    args = ap.parse_args()

    global video_train_feat_path
    global video_test_feat_path
    global outfile 

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

