import pickle
import numpy as np

def to_one_hot(labels, num_classes=48):
    """Convert class labels from scalars to one-hot vectors."""
    
    num_labels = len(labels)
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels] = 1
    return labels_one_hot

f = open('./data/mfcc_data/train_data.pkl', 'rb')
data = pickle.load(f)
print(len(data))

data_p = [np.lib.pad(s, ((0,777-len(s)),(0,0)), 'constant' , constant_values=(0)) for s in data]
print(len(data_p))

d = np.array(data_p)
print(d.shape)
#print(d[0][0])

np.save('./data/train_data.npy', d)



f3 = open('./data/mfcc_data/test_data.pkl', 'rb')
data = pickle.load(f3)

data_p = [np.lib.pad(s, ((0,777-len(s)),(0,0)), 'constant' , constant_values=(0)) for s in data]
print(len(data_p))

d = np.array(data_p)
print(d.shape)
#print(d[0][0])

np.save('./data/test_data.npy', d)




f2 = open('./data/mfcc_data/train_mapped_label.pkl', 'rb')
label = pickle.load(f2)

label_one_hot = np.array([to_one_hot(i) for i in label])
label_one_hot = np.array([np.lib.pad(s, ((0,777-len(s)),(0,0)), 'constant' , constant_values=(0)) for s in label_one_hot])
print(label_one_hot.shape)
#print(label_one_hot[0][0])

np.save('./data/train_label.npy', label_one_hot)

