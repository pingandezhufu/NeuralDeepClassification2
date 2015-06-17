from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from keras.utils import np_utils, generic_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Reshape
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
        print(X.shape)
        print("Labels are:")
        print(labels)
        return X, labels
    else:
        X = X[:,0:-1].astype(np.float32)
        print("X is:")
        print(X)
        return X

def load_data_4D(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X) # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, 0]
        temp=np.zeros((1,42000,1,783))
        X = preprocess_data(X)
        channel=[]
        samples=[]
        width=[]
        height=[]
        temp=np.array
        for i in range (42000):
            print(i)
            for j in range (1):
                for z in range (783):
                    tempSample=X[i]
                    tempValue=tempSample[z]
                    height.append(tempValue)
                width.append(height)
            samples.append(height)
        channel.append(samples)
        print(channel.ndim)
        X=np.asarray(channel)
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
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))


print("Loading data...")
X, labels = load_data_4D('trainMnist.csv', train=True)
data=np.array(X)
print("Data for np:")
print(data.shape)
print(data.ndim)
y, encoder = preprocess_labels(labels)

X_test = load_data('testMnist.csv', train=False)
X_test = preprocess_data(X_test)

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')

print("Building model...")
model=Sequential()
neurons=2048
model = Sequential()
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
model.add(Dense(64*8*8, 256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256, 10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)

print("Training model...")
model.fit(X, y, nb_epoch=40, batch_size=1, validation_split=0.11,verbose=1)


print("Generating submission...")

proba = model.predict_proba(X_test)

make_submission(proba, encoder, fname='Submission11.csv')
