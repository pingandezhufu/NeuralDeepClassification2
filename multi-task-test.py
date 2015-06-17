from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD
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
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def preprocess_labels(y, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))


print("Loading data...")
X, labels = load_data('train.csv', train=True)
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(labels)

X_test, ids = load_data('test.csv', train=False)
X_test, _ = preprocess_data(X_test, scaler)

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')

print("Building model...")
neurons=64
outputNeurons=1
model= Sequential()
# left = Sequential()
# left.add(Dense(dims, 50))
# left.add(Activation('relu'))
#
# right = Sequential()
# right.add(Dense(dims, 50))
# right.add(Activation('relu'))

# model = Sequential()
# model.add(Merge([left, right], mode='sum'))

# model.add(Dense(50, nb_classes))
# model.add(Activation('softmax'))
tasks=[]
inputSet=[]
numTasks=9
for i in range (numTasks):
    model1 = Sequential()
    model1.add(Dense(dims, neurons, init='glorot_uniform'))
    model1.add(Activation('relu'))
    model1.add(BatchNormalization((neurons,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(neurons, neurons, init='glorot_uniform'))
    model1.add(Activation('relu'))
    model1.add(BatchNormalization((neurons,)))
    model1.add(Dropout(0.5))

    # model1.add(Dense(neurons, neurons, init='glorot_uniform'))
    # model1.add(Activation('relu'))
    # model1.add(BatchNormalization((neurons,)))
    # model1.add(Dropout(0.5))

    model1.add(Dense(neurons, outputNeurons, init='glorot_uniform'))
    model1.add(Activation('relu'))
    model1.add(BatchNormalization((outputNeurons,)))
    model1.add(Dropout(0.5))
    tasks.append(model1)
    inputSet.append(X)

model.add(Merge(tasks, mode='concat'))

model.add(Dense(len(tasks)*outputNeurons, nb_classes))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.compile(loss='categorical_crossentropy', optimizer='adam')

print("Training model...")
model.fit(inputSet, y, nb_epoch=30, batch_size=16,verbose=1, validation_split=0.11, show_accuracy=True)


print("Generating submission...")

proba =model.predict_proba(X_test)

make_submission(proba, ids, encoder, fname='Submission11.csv')