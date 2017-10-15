import pickle
import numpy as np
import pdb
import preprocessing as pp

NUM_CLASSES = 48

def load_dataset(data_pickle, 
                 labels_pickle, 
                 to_one_hot=True, 
                 batch_size=6,
                 normalization=True):
    
    with open(data_pickle, 'rb') as data_dump:
        data_sentences = pickle.load(data_dump)
        
    with open(labels_pickle, 'rb') as labels_dump:
        labels_sentences = pickle.load(labels_dump)
    
    # Normalize the mfccs
    if normalization:
        print("Normalizing")
        data = [pp.normalize_mfcc(s) for s in data_sentences]
    else:
        data = data_sentences

    # Possibly flatten the sentences
    if to_one_hot:
        labels = [pp.to_one_hot(labels_scalar, NUM_CLASSES) for labels_scalar in labels_sentences]

    print("Preprocessing done")
    eval = int(len(data) / 5)
    #return DataSet(np.array(data), np.array(labels), batch_size)    
    return DataSet(np.array(data[eval:]), np.array(labels[eval:]), batch_size), DataSet(np.array(data[:eval]), np.array(labels[:eval]), batch_size)


def load_test_dataset(data_pickle, 
                 batch_size=1,
                 normalization=True):
    
    with open(data_pickle, 'rb') as data_dump:
        data_sentences = pickle.load(data_dump)
    
    # Normalize the mfccs
    if normalization:
        print("Normalizing")
        data = [pp.normalize_mfcc(s) for s in data_sentences]
    else:
        data = data_sentences

    print("Preprocessing done")
    return DataSet(np.array(data), None, batch_size)


class DataSet(object):
    def __init__(self, data, labels, batch_size=6):
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = data.shape[0]
        # required for next_sequence_batch
        self._batch_size = batch_size
        
    @property
    def data(self):
        return self._data
        
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed
        
    @property
    def batch_size(self):
        return self._batch_size
    
    def reset_epoch(self, batch_size):
        self._batch_size = batch_size
        # Shuffle the data
        perm = np.arange(len(self._data))
        np.random.shuffle(perm)
        self._data = self._data[perm]
        self._labels = self._labels[perm]
    
    def next_batch(self, batch_size, _pad=False):

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(len(self._data))
            np.random.shuffle(perm)
            #print("End of epoch")
            self._data = self._data[perm]
            self._labels = self._labels[perm]
            self._index_in_epoch = 0                 
            # Return Nones to signal the training loop
            return None, None, None
        
        end = self._index_in_epoch
        batch_data = []
        batch_label = []
        seq_len = []
        max_seq_len = 1

        for s in self._data[start:end]:
            max_seq_len = max(s.shape[0], max_seq_len)

        p = 0

        if(_pad):
            p=1
            
        for s in self._data[start:end]:
            sl = s.shape[0]
            seq_len.append(sl)
            batch_data.append(np.lib.pad(s, ((p,max_seq_len-sl+p),(0,0)), 'edge'))
        
        for s in self._labels[start:end]:
            sl = s.shape[0]
            batch_label.append(np.lib.pad(s, ((0,max_seq_len-sl),(0,0)), 'edge'))

        return np.array(batch_data), np.array(batch_label), seq_len


    def next_test_batch(self, batch_size, _pad=False):

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            self._index_in_epoch = 0
            # Return Nones if end of testing
            return None, None, None
        
        end = self._index_in_epoch
        batch_data = []
        seq_len = []
        max_seq_len = 0
        for s in self._data[start:end]:
            max_seq_len = max(s.shape[0], max_seq_len)

        p = 0

        if(_pad):
            p=1
            
        for s in self._data[start:end]:
            sl = s.shape[0]
            seq_len.append(sl)
            batch_data.append(np.lib.pad(s, ((p,max_seq_len-sl+p),(0,0)), 'edge'))

        return np.array(batch_data), seq_len