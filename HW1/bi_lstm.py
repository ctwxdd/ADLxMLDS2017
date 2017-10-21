import numpy as np
import sys
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Masking
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
np.random.seed(7)
#import LD

TIME_STEP=777
INPUT_SIZE=39
BATCH_SIZE=16
NUM_CLASS=48
EPOCH=2

#x_test = LD.load_data('data', 'mfcc', 'test')
#np.save('data/mfcc/test.npy', x_test)
#x_train = LD.load_data('data', 'mfcc', 'train')
#np.save('data/mfcc/train.npy', x_train)
#y_train = LD.load_label('data', 'align_train.lab')
#np.save('data/label/label.npy', y_train)

if sys.argv[1] == 'fbank':
    INPUT_SIZE = 69
    x_train = np.load('data/fbank/train.npy')
    x_test = np.load('data/fbank/test.npy')
else:
    x_train = np.load('data/train_data.npy')
    x_test = np.load('data/test_data.npy')
y_train = np.load('data/train_label.npy')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

x_val = x_train[:16]
y_val = y_train[:16]

x_train = x_train[16:]
y_train = y_train[16:]

model = Sequential()
model.add(Masking(mask_value=0., input_shape=(BATCH_SIZE, TIME_STEP, INPUT_SIZE)))
model.add(Bidirectional(LSTM(64, return_sequences=True), batch_input_shape=(BATCH_SIZE, TIME_STEP, INPUT_SIZE)))

# model.add(Bidirectional(LSTM(64, return_sequences=True, stateful=True)))
# model.add(Bidirectional(LSTM(64, return_sequences=True, stateful=True)))
# model.add(Bidirectional(LSTM(64, return_sequences=True, stateful=True)))
# model.add(Bidirectional(LSTM(64, return_sequences=True, stateful=True)))
# model.add(Bidirectional(LSTM(64, return_sequences=True, stateful=True)))

model.add(Dense(48, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))

model_json = model.to_json()
with open("model/model_bi_f.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model/model_bi_f.h5")
print("Saved model to disk")

# Final evaluation of the model
#pred = model.predict(x_test, batch_size=16)
#sssnp.save('bi_fbank.npy', pred)

#scores = model.evaluate(x_val, y_val, batch_size=1)
#print("Accuracy: %.2f%%" % (scores[1]*100))
