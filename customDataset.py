from __future__ import absolute_import
#from keras.datasets.cifar import load_batch
from loadCustomCifar100 import load_batch
from keras.datasets import cifar
from keras.datasets.data_utils import get_file
import numpy as np
import os
from numpy import *
def load_data(label_mode='fine'):
    dirname = "cifar-10-batches-py"
    origin = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    path = get_file(dirname, origin=origin, untar=True)

    nb_test_samples = 10000
    nb_train_samples = 50000

    X_train2 = np.zeros((nb_train_samples, 3, 32, 32), dtype="uint8")
    y_train2 = np.zeros((nb_train_samples,), dtype="uint8")

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = cifar.load_batch(fpath)
        X_train2[(i-1)*10000:i*10000, :, :, :] = data
        y_train2[(i-1)*10000:i*10000] = labels

    fpath = os.path.join(path, 'test_batch')
    X_test2, y_test2 = cifar.load_batch(fpath)

    y_train2 = np.reshape(y_train2, (len(y_train2), 1))
    y_test2 = np.reshape(y_test2, (len(y_test2), 1))
    ################################################################
    if label_mode not in ['fine', 'coarse']:
        raise Exception('label_mode must be one of "fine" "coarse".')

    dirname = "cifar-100-python"
    origin = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    path = get_file(dirname, origin=origin, untar=True)

    nb_test_samples = 500
    nb_train_samples = 2500

    fpath = os.path.join(path, 'train')
    X_train1, y_train1 = load_batch(fpath, label_key=label_mode+'_labels')

    fpath = os.path.join(path, 'test')
    X_test1, y_test1 = load_batch(fpath, label_key=label_mode+'_labels')

    y_train1 = np.reshape(y_train1, (len(y_train1), 1))
    y_test1 = np.reshape(y_test1, (len(y_test1), 1))

    #####################################################################
    print(type(X_train1))
    print(type(X_train2))
    X_train=X_train1.tolist()+X_train2.tolist()
    print("X_train transformation worked")
    X_test=X_test1.tolist()+X_test2.tolist()
    print("X_test transformation worked")
    X_test=asarray(X_test)
    print("X_test revertion worked")
    X_train=asarray(X_train)
    print("X_train revertion worked")
    print(type(y_test1))
    print(type(y_test2))
    y_test=y_test1.tolist()+y_test2.tolist()
    y_train=y_train1.tolist()+y_train2.tolist()
    y_test=asarray(y_test)
    y_train=asarray(y_train)

    nb_test_samples=len(X_test)
    print(nb_test_samples)
    nb_train_samples=len(X_train)
    print(nb_train_samples)
    return (X_train, y_train), (X_test, y_test)
