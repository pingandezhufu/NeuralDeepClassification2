from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

'''
    This demonstrates how to reach a score of 0.4890 (local validation)
    on the Kaggle Otto challenge, with a deep net using Keras.
    Compatible Python 2.7-3.4
    Recommended to run on GPU:
        Command: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python kaggle_otto_nn.py
        On EC2 g2.2xlarge instance: 19s/epoch. 6-7 minutes total training time.
    Best validation score at epoch 21: 0.4881
    Try it at home:
        - with/without BatchNormalization (BatchNormalization helps!)
        - with ReLU or with PReLU (PReLU helps!)
        - with smaller layers, largers layers
        - with more layers, less layers
        - with different optimizers (SGD+momentum+decay is probably better than Adam!)
'''

np.random.seed(1337) # for reproducibility

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X) # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, 0]
        print("X is :")
        print(X)
        print("Labels are:")
        print(labels)
        return X, labels
    else:
        X = X[:,0:-1].astype(np.float32)
        print("X is:")
        print(X)
        return X

def preprocess_data(X):
    X = X/255
    return X

def preprocess_labels(y, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def make_submission(y_prob, encoder, fname):
    with open(fname, 'w') as f:
        f.write("Classes")
        f.write('\n')
        for i in range (len(y_prob)):
            f.write(y_prob(i))
            print(y_prob(i))
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))


print("Loading data...")
X, labels = load_data('trainMnist.csv', train=True)
X = preprocess_data(X)
y, encoder = preprocess_labels(labels)

X_test = load_data('testMnist.csv', train=False)
X_test = preprocess_data(X_test)

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')

print("Building model...")
model=Sequential()
//neurons=2048*2.5
model.add(Convolution2D(32, 3, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 32, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64*8*8, 512, init='normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(dims, neurons, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(BatchNormalization((neurons,)))
model.add(Dropout(0.5))

model.add(Dense(neurons, neurons/2, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(BatchNormalization((neurons/2,)))
model.add(Dropout(0.5))

model.add(Dense(neurons/2, neurons/4, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(BatchNormalization((neurons/4,)))
model.add(Dropout(0.5))

model.add(Dense(neurons/4, neurons/8, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(BatchNormalization((neurons/8,)))
model.add(Dropout(0.5))

model.add(Dense(neurons/8, nb_classes, init='glorot_uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)

print("Training model...")
model.fit(X, y, nb_epoch=10, batch_size=32, validation_split=0.11,show_accuracy=True,verbose=1)


print("Generating submission...")

proba = model.predict_proba(X_test)

make_submission(proba, encoder, fname='Submission11.csv')
