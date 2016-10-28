import numpy as np
from abc import ABCMeta, abstractmethod


class LinearClassifier(object):
	__metaclass__ = ABCMeta

	def __init__(self, C = 1, Iter = 1000, reportIter = 10):
		self.C = C
		self.Iter = Iter
		self.reportIter = reportIter

	def adjustVectors(self, X):
		return np.concatenate((X, np.ones((len(X), 1))), axis = 1)

	@abstractmethod
	def adjustLabels(self, Y):
		pass

	def fit(self, X, Y):
		self.dimension = len(X[0]) + 1
		self.weight = np.random.random(self.dimension)
		train_X = self.adjustVectors(X)
		train_Y = self.adjustLabels(Y)
		self.learningCurve = []
		for i in xrange(self.Iter):
			self.weight -= self.learningRate(i) * self.gradient(train_X, train_Y)
			if i % self.reportIter == 0:
				loss = self.evaluate(X, Y)
				self.learningCurve.append(loss)
				# print "Epoch %d, loss %f" % (i, loss)
				# print self.weight
		return self.learningCurve


	@abstractmethod
	def learningRate(self, step):
		pass

	def gradient(self, train_X, train_Y):
		gradient = np.zeros(self.dimension)
		for i in xrange(len(train_X)):
			gradient += self.gradientAt(train_X[i], train_Y[i])
		return gradient

	@abstractmethod
	def gradientAt(self, X, Y):
		pass

	def evaluate(self, X, Y):
		test_X = self.adjustVectors(X)
		test_Y = self.adjustLabels(Y)
		loss = 0.0
		for i in xrange(len(test_X)):
			if test_Y[i] != self.evaluateAt(test_X[i]):
				loss += 1.0
		return loss / len(test_Y)

	@abstractmethod
	def evaluateAt(self, X):
		pass



class LinearRegression(LinearClassifier):
	def adjustLabels(self, Y):
		new_Y = np.copy(Y)
		new_Y[Y == 0] = -1
		return new_Y

	def learningRate(self, step):
		return 1e-6

	def gradientAt(self, X, Y):
		return X * (2.0 * (np.dot(self.weight, X) - Y))

	def evaluateAt(self, X):
		return 1 if np.dot(self.weight, X) >= 0 else -1


class LogisticRegression(LinearClassifier):
	def adjustLabels(self, Y):
		return np.copy(Y)

	def learningRate(self, step):
		return 1e-2

	def g(self, X):
		return 1.0 / (1.0 + np.exp(-np.dot(self.weight, X)))

	def gradientAt(self, X, Y):
		g = self.g(X)
		return X * (2.0 * (g - Y) * g * (1.0 - g))

	def evaluateAt(self, X):
		return 1 if self.g(X) >= 0.5 else 0


class Perceptron(LinearClassifier):
	def adjustLabels(self, Y):
		new_Y = np.copy(Y)
		new_Y[Y == 0] = -1
		return new_Y

	def learningRate(self, step):
		return 1e-6

	def gradientAt(self, X, Y):
		if np.dot(self.weight, X) * Y > 0.0:
			return np.zeros(self.dimension)
		return X * (-Y)

	def evaluateAt(self, X):
		return 1 if np.dot(self.weight, X) >= 0.0 else -1


class LinearSVM(LinearClassifier):
	def adjustLabels(self, Y):
		new_Y = np.copy(Y)
		new_Y[Y == 0] = -1
		return new_Y

	def learningRate(self, step):
		return 1e-6 / self.C

	def gradient(self, train_X, train_Y):
		gradient = self.weight * 2
		for i in xrange(len(train_X)):
			gradient += self.C * self.gradientAt(train_X[i], train_Y[i])
		return gradient

	def gradientAt(self, X, Y):
		if np.dot(self.weight, X) * Y > 1.0:
			return np.zeros(self.dimension)
		return X * (-Y)

	def evaluateAt(self, X):
		return 1 if np.dot(self.weight, X) >= 0.0 else -1
		

