import time
import numpy as np
from numpy import linalg as LA
from abc import ABCMeta, abstractmethod


class NeuralNetwork(object):
	def __init__(self, hidden_layers, dimensions):
		assert len(dimensions) == len(hidden_layers) + 1
		self.dimensions = dimensions
		self.weights = [(np.random.rand(dimensions[i + 1], dimensions[i] + 1) * 2.0 - 1.0) * 1e-2 for i in xrange(len(dimensions) - 1)]
		self.layers = [hidden_layers[i](dimensions[i + 1]) for i in xrange(len(hidden_layers))]
		self.time = 0.0

	def fit(self, train_X, train_Y, iteration, report_iteration, batch_size):
		assert len(train_X) == len(train_Y)
		self.labels = np.unique(train_Y)
		self.weights.append((np.random.rand(len(self.labels), self.dimensions[-1] + 1) * 2.0 - 1.0) * 1e-2)
		self.layers.append(Softmax(self.labels))

		assert len(self.weights) == len(self.layers)

		for i in xrange(iteration):
			batch_idx = np.random.permutation(range(len(train_X)))[:batch_size]
			batch_X, batch_Y = train_X[batch_idx], train_Y[batch_idx]
			gradients = self.back_propagation(batch_X, batch_Y)
			self.update(gradients, i, batch_size)

			if i % report_iteration == 0:
				print self.test(train_X, train_Y)
		print self.time / iteration

	def update(self, gradients, step, batch_size):
		assert len(self.weights) == len(gradients)
		for i in xrange(len(gradients)):
			self.weights[i] -= self.learning_rate(step) * gradients[i] / batch_size

	def learning_rate(self, step):
		return 1e-1 / (1 + step * 1e-4)

	def back_propagation(self, batch_X, batch_Y):
		assert len(batch_X) == len(batch_Y)
		gradients = [np.zeros(weight.shape) for weight in self.weights]

		start = time.time()
		for i in xrange(len(batch_X)):
			# forward evaluation
			X = [None for layer in self.layers]
			Y = [None for j in xrange(len(self.layers) + 1)]
			Y[0] = batch_X[i]

			for j in xrange(len(self.layers)):
				X[j] = np.dot(self.weights[j], np.append(Y[j], 1))
				Y[j + 1] = self.layers[j].evaluate(X[j])

			# back propagation
			jacobian = [None for layer in self.layers]
			jacobian[-1] = self.layers[-1].jacobian(X[-1], batch_Y[i])
			for j in xrange(len(jacobian) - 2, -1, -1):
				jacobian[j] = np.dot(self.layers[j].jacobian(X[j], Y[j + 1]), np.dot(self.weights[j + 1][:, :-1].T, jacobian[j + 1]))

			gradient = [None for weight in self.weights]
			for j in xrange(len(gradient) - 1, -1, -1):
				gradient[j] = np.outer(jacobian[j], np.append(Y[j], 1))

			for j in xrange(len(gradients)):
				gradients[j] += gradient[j]
		self.time += time.time() - start

		return gradients

	def test(self, test_X, test_Y):
		loss = 0.0
		predict_Y = self.predict(test_X)
		for i in xrange(len(predict_Y)):
			if predict_Y[i] != test_Y[i]:
				loss += 1.0
		return loss / len(test_Y)

	def predict(self, test_X):
		return np.array([self.evaluate(x) for x in test_X])

	def evaluate(self, X):
		Y = np.copy(X)
		for i in xrange(len(self.layers)):
			Y = self.layers[i].evaluate(np.dot(self.weights[i], np.append(Y, 1)))
		return Y


class Layer(object):
	@abstractmethod
	def jacobian(self, X, Y):
		pass

	@abstractmethod
	def evaluate(self, X):
		pass


class Softmax(Layer):
	def __init__(self, labels):
		self.num_unit = len(labels)
		self.labels = labels

	def jacobian(self, X, Y):
		idx = np.where(self.labels == Y)[0]
		G = self.G(X)
		return np.array([G[i] - int(i == idx) for i in xrange(len(X))])

	def G(self, X):
		G = np.array([np.exp(x) for x in X])
		return np.array([g / np.sum(G) for g in G])

	def evaluate(self, X):
		return self.labels[np.argmax(X)]


class HiddenLayer(Layer):
	def __init__(self, num_unit):
		self.num_unit = num_unit

	def jacobian(self, X, Y):
		return np.diag(np.array([self.gprime(g) for g in Y]))

	@abstractmethod
	def gprime(self, g):
		pass

	def evaluate(self, X):
		return np.array([self.g(x) for x in X])

	@abstractmethod
	def g(self, x):
		pass


class ReLU(HiddenLayer):
	def gprime(self, g):
		return 1 if g >= 0 else 0

	def g(self, x):
		return x if x >= 0 else 0


class Sigmoid(HiddenLayer):
	def gprime(self, g):
		return g * (1.0 - g)

	def g(self, x):
		return 1.0 / (1.0 + np.exp(-x))


class Tanh(HiddenLayer):
	def gprime(self, g):
		return 1.0 - np.square(g)

	def g(self, x):
		return np.tanh(x)


