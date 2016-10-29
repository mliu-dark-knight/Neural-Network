import numpy as np


class RBM(object):
	def __init__(self, v_dimension, h_dimension):
		self.v_dimension = v_dimension
		self.h_dimension = h_dimension
		self.W = np.random.rand(self.v_dimension, self.h_dimension) * 2.0 - 1.0
		self.b = np.random.rand(self.v_dimension) * 2.0 - 1.0
		self.c = np.random.rand(self.h_dimension) * 2.0 - 1.0
	
	def fit(self, Vs, iter=10, batch_size=10):
		self.Hs = np.random.randint(2, size=(len(Vs), self.h_dimension))
		for i in xrange(iter):
			sample = Vs[np.random.permutation(range(len(Vs)))[:batch_size]]
			dW, db, dc = self.gradient(sample)
			learning_rate = self.learning_rate(i)
			self.W += learning_rate * dW / batch_size
			self.b += learning_rate * db / batch_size
			self.c += learning_rate * dc / batch_size

	def learning_rate(self, step):
		return 1e-1 / (1.0 + step)

	def gradient(self, Vs):
		dWs = np.zeros(self.W.shape)
		dbs = np.zeros(self.b.shape)
		dcs = np.zeros(self.c.shape)
		for V in Vs:
			dW, db, dc = self.gradient_at(V)
			dWs += dW
			dbs += db
			dcs += dc
		return dWs, dbs, dcs

	def gradient_at(self, V):
		H = self.v_to_h(V)
		E_V, E_H = self.contrastive_divergence(1)
		dW = np.outer(V.T, H) - np.outer(E_V.T, E_H)
		db = V - E_V
		dc = H - E_H
		return dW, db, dc

	def contrastive_divergence(self, k):
		V = np.random.randint(2, size=(self.v_dimension))
		for i in xrange(k):
			H = self.v_to_h(V)
			V = self.h_to_v(H)
		return V, H

	def sigmoid(self, v):
		return 1.0 / (1.0 + np.exp(-v))

	def h_to_v(self, H):
		return np.array([self.sigmoid(h) for h in np.dot(self.W, H) + self.b])

	def v_to_h(self, V):
		return np.array([self.sigmoid(v) for v in np.dot(V.T, self.W) + self.c])

	def inference(self, V):
		return self.v_to_h(V)


