import os
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from multiprocessing import Process

trainX_dir = "train/mfc"
trainY_file = "train/lab/hw1train_labels.txt"
devX_dir = "dev/mfc"
devY_file = "dev/lab/hw1dev_labels.txt"
testX_dir = "test/mfc"
testY_file = "test/lab/hw1test_labels.txt"
iteration = 10000
report_iteration = 100


def invalid(vector):
	return np.isnan(vector).any() or np.isinf(vector).any()


class LinearClassifier(object):
	__metaclass__ = ABCMeta

	def __init__(self, train_X, train_Y, dev_X, dev_Y, Iter, reportIter):
		self.train_X = self.adjustVectors(train_X)
		self.train_Y = self.adjustLabels(train_Y)
		self.dev_X = self.adjustVectors(dev_X)
		self.dev_Y = self.adjustLabels(dev_Y)
		self.Iter = Iter
		self.reportIter = reportIter
		self.dimension = len(train_X[0]) + 1
		self.weight = np.random.random(self.dimension) * 2.0 - 1.0

	def adjustVectors(self, X):
		return np.concatenate((X, np.ones((len(X), 1))), axis = 1)

	@abstractmethod
	def adjustLabels(self, Y):
		pass

	def fit(self):
		self.learningCurve = [self.evaluate(self.train_X, self.train_Y)]
		for i in xrange(self.Iter):
			self.weight -= self.learningRate(i) * self.gradient()
			if (i + 1) % self.reportIter == 0:
				loss = self.evaluate(self.train_X, self.train_Y)
				self.learningCurve.append(loss)
				print "Epoch %d, loss %f" % (i, loss)
				print self.weight
		return self.learningCurve


	@abstractmethod
	def learningRate(self, step):
		pass

	def gradient(self):
		gradient = np.zeros(self.dimension)
		for i in xrange(len(self.train_X)):
			if invalid(self.train_X[i]):
				continue
			gradient += self.gradientAt(self.train_X[i], self.train_Y[i])
		return gradient

	@abstractmethod
	def gradientAt(self, X, Y):
		pass

	def evaluate(self, test_X, test_Y):
		loss = 0.0
		for i in xrange(len(test_X)):
			if invalid(test_X[i]) or test_Y[i] != self.evaluateAt(test_X[i]):
				loss += 1.0
		return loss / len(test_Y)

	@abstractmethod
	def evaluateAt(self, X):
		pass



class LinearRegression(LinearClassifier):
	def adjustLabels(self, Y):
		return np.copy(Y)

	def learningRate(self, step):
		return 1e-6

	def gradientAt(self, X, Y):
		return X * (2.0 * (np.dot(self.weight, X) - Y))

	def evaluateAt(self, X):
		return 1 if np.dot(self.weight, X) >= 0.5 else 0


class LogisticRegression(LinearClassifier):
	def adjustLabels(self, Y):
		return np.copy(Y)

	def learningRate(self, step):
		return 1e-4 / (1 + step)

	def gradientAt(self, X, Y):
		dot = np.dot(self.weight, X)
		return X * (2.0 * (dot - Y) * dot * (1.0 - dot))

	def evaluateAt(self, X):
		return 1 if 1.0 / (1.0 + np.exp(-np.dot(self.weight, X))) >= 0.5 else 0


class Perceptron(LinearClassifier):
	def adjustLabels(self, Y):
		new_Y = np.copy(Y)
		new_Y[Y == 0] = -1
		return new_Y

	def learningRate(self, step):
		return 1e-4 / (1 + step)

	def gradientAt(self, X, Y):
		if np.dot(self.weight, X) * Y > 0:
			return np.zeros(self.dimension)
		return X * (-Y)

	def evaluateAt(self, X):
		return 1 if np.dot(self.weight, X) >= 0.0 else -1


class LinearSVM(LinearClassifier):
	def adjustLabels(self, Y):
		pass

	def learningRate(self, step):
		pass

	def gradientAt(self, X, Y):
		pass

	def evaluateAt(self, X):
		pass
		


def read_X(directory):
	X = []
	for filename in os.listdir(directory):
		if filename == ".DS_Store":
			continue
		X.append(np.array(map(float, open(directory + '/' + filename, 'r').readline().split())))
	return np.array(X)


def read_Y(filename):
	Y = []
	with open(filename, 'r') as f:
		for line in f:
			Y.append(int(line[:-1].split()[0]))
	return np.array(Y)


def main():
	train_X = read_X(trainX_dir)
	train_Y = read_Y(trainY_file)
	dev_X = read_X(devX_dir)
	dev_Y = read_Y(devY_file)
	# linear_regression = LinearRegression(train_X, train_Y, dev_X, dev_Y, iteration, report_iteration)
	# linearRegressionLearningCurve = linear_regression.fit()
	# logistic_regression = LogisticRegression(train_X, train_Y, dev_X, dev_Y ,iteration, report_iteration)
	# logisticRegressionLearningCurve = logistic_regression.fit()
	perceptron = Perceptron(train_X, train_Y, dev_X, dev_Y, iteration, report_iteration)
	perceptronLearningCurve = perceptron.fit()
	# plt.plot(range(1, iteration + 1, report_iteration), linearRegressionLearningCurve, 'r', label = "linear regression")
	# plt.plot(range(1, iteration + 1, report_iteration), logisticRegressionLearningCurve, 'g', label = "logistic regression")
	plt.plot(range(1, iteration + 1, report_iteration), perceptronLearningCurve, 'b', label = "perceptron")
	plt.legend(loc = 1)
	plt.show()



main()

