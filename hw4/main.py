import os
import struct
import numpy as np
from RBM import RBM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/")

def test_RBM(train_X, train_Y, test_X, test_Y):
    #Create and train RBM
    rbm = RBM(28 * 28, 200)
    rbm.fit(train_X)

    # train_X = np.array([rbm.inference(x) for x in train_X])
    # test_X = np.array([rbm.inference(x) for x in test_X])

    # logreg = LogisticRegression(max_iter=10)
    # logreg.fit(train_X, train_Y)

    # predict_Y = logreg.predict(train_X)
    # print("Accuracy on training data")
    # print(accuracy_score(train_Y, predict_Y))
    # predict_Y = logreg.predict(test_X)
    # print("Accuracy on test data")
    # print(accuracy_score(test_Y, predict_Y))

def to_binary(X):
    """Returns a binary version of the input image"""
    X = X.reshape((-1, 28 * 28))
    X[X > 0] = 1
    return X


def main():
    #Extract data
    train_X = mnist.train.images
    train_X = to_binary(train_X)
    print(train_X.shape)
    train_Y = mnist.train.labels
    print(train_Y.shape)
    # test_X, test_Y = read(testX_file, testY_file)
    # test_X = to_binary(test_X)
    test_X = mnist.test.images
    test_X = to_binary(test_X)
    test_Y = mnist.test.labels

    #Train and test RBM
    test_RBM(train_X, train_Y, test_X, test_Y)
    

main()
