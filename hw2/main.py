import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from NeuralNetwork import *

trainX_dir = "train/fbc/"
trainY_file = "train/lab/hw2train_labels.txt"
testX_dir = "eval/fbc/"
testY_file = "eval/lab/hw2eval_labels.txt"
iteration = 200000
report_iteration = 20000
frame_limit = 16 * 70
batch_size = 10


def invalid(vector):
	return np.isnan(vector).any() or np.isinf(vector).any() or len(vector) < frame_limit

def read_single_X(file):
	X = []
	with open(file, 'r') as f:
		for line in f:
			X.extend(list(map(float, line.rstrip().split())))
	return np.array(X)



def read_XY(directory, filename):
	X, Y = {}, {}
	for file in os.listdir(directory):
		if file == ".DS_Store":
			continue
		x = read_single_X(directory + file)
		if not invalid(x):
			X[directory + file] = x[:frame_limit]

	with open(filename, 'r') as f:
		for line in f:
			line = line[:-1].split()
			Y[line[1]] = int(line[0])

	xs, ys = [], []
	for file in X:
		if file in Y:
			xs.append(X[file])
			ys.append(Y[file])

	return np.array(xs).astype(np.float64), np.array(ys).astype(np.float64)


def main():
	train_X, train_Y = read_XY(trainX_dir, trainY_file)
	test_X, test_Y = read_XY(testX_dir, testY_file)

	for activation in [Sigmoid, Tanh, ReLU]:
		for num_unit in [50, 10]:
			nn = NeuralNetwork([activation, activation], [frame_limit, num_unit, num_unit])
			nn.fit(train_X, train_Y, iteration, report_iteration, batch_size)
			predict_Y = nn.predict(train_X)
			print "Accuracy on training data"
			print accuracy_score(train_Y, predict_Y)
			print "Confusion matrix on training data"
			print confusion_matrix(train_Y, predict_Y)

			predict_Y = nn.predict(test_X)
			print "Accuracy on test data"
			print accuracy_score(test_Y, predict_Y)
			print "Confusion matrix on test data"
			print confusion_matrix(test_Y, predict_Y)


main()

