import os
import struct
import numpy as np
from RBM import RBM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

trainX_file = "mnist/train-images.idx3-ubyte"
trainY_file = "mnist/train-labels.idx1-ubyte"
testX_file = "mnist/t10k-images.idx3-ubyte"
testY_file = "mnist/t10k-labels.idx1-ubyte"


def read(fname_img, fname_lbl):
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    flbl.close()

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    fimg.close()
    
    return img, lbl

def test_RBM(train_X, train_Y, test_X, test_Y):
    rbm = RBM(28 * 28, 200)
    rbm.fit(train_X)

    train_X = np.array([rbm.inference(x) for x in train_X])
    test_X = np.array([rbm.inference(x) for x in test_X])

    logreg = LogisticRegression(max_iter=10)
    logreg.fit(train_X, train_Y)

    predict_Y = logreg.predict(train_X)
    print "Accuracy on training data"
    print accuracy_score(train_Y, predict_Y)
    predict_Y = logreg.predict(test_X)
    print "Accuracy on test data"
    print accuracy_score(test_Y, predict_Y)

def to_binary(X):
    X = X.reshape((-1, 28 * 28))
    X[X > 0] = 1
    return X


def main():
    train_X, train_Y = read(trainX_file, trainY_file)
    train_X = to_binary(train_X)
    test_X, test_Y = read(testX_file, testY_file)
    test_X = to_binary(test_X)
    test_RBM(train_X, train_Y, test_X, test_Y)

main()
