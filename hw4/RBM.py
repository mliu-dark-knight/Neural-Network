import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class RBM(object):
	def __init__(self, v_dimension, h_dimension):
		self.v_dimension = v_dimension # 784
		self.h_dimension = h_dimension # 200
		self.W = np.random.rand(self.v_dimension, self.h_dimension) * 2.0 - 1.0 # (784, 200)
		self.b = np.random.rand(self.v_dimension) * 2.0 - 1.0 # (784,)
		self.c = np.random.rand(self.h_dimension) * 2.0 - 1.0 # (200,)
	
	def fit(self, Vs, iter=100, batch_size=100):
		self.Vs = Vs
		for i in range(iter):	
			sample = Vs[np.random.permutation(range(len(Vs)))[:batch_size]] # Vs = (10, 784)
			dW, db, dc = self.gradient(sample)
			learning_rate = self.learning_rate(i)
			self.W += (learning_rate * dW / batch_size)
			self.b += (learning_rate * db / batch_size)
			self.c += (learning_rate * dc / batch_size)

	def learning_rate(self, step):
		return 1e-1 / (1.0 + step * 1e-4)

	def gradient(self, Vs):
		dWs = np.zeros(self.W.shape)
		dbs = np.zeros(self.b.shape)
		dcs = np.zeros(self.c.shape)
		# Sum over all training tokens in batch
		for V in Vs:
			dW, db, dc = self.gradient_at(V)
			dWs += dW
			dbs += db
			dcs += dc
		return dWs, dbs, dcs

	def gradient_at(self, V):
		H = self.v_to_h(V)
		E_V, E_H = self.contrastive_divergence(1)
		# Implement gradient quations from CFS
		dW = np.outer(V, H) - np.outer(E_V, E_H)
		db = V - E_V
		dc = H - E_H
		return dW, db, dc

	def contrastive_divergence(self, k):
		V = self.Vs[np.random.randint(len(self.Vs), size=1)[0]]
		for i in range(k - 1):
			H = self.v_to_h(V)
			V = self.h_to_v(H)
		H = self.v_to_h(V)
		return V, H

	def sigmoid(self, v):
		return 1.0 / (1.0 + np.exp(-v))

	def h_to_v(self, H):
		"""Computes the expectation of V|H """
		return np.array([self.sigmoid(h) for h in np.dot(self.W, H) + self.b])

	def v_to_h(self, V):
		"""Computes the expectation of H|V"""
		return np.array([self.sigmoid(v) for v in np.dot(V.T, self.W) + self.c])

	def inference(self, V):
		"""Feed forward neural net"""
		return self.v_to_h(V)

	def get_filters(self):
		W = self.W.T
		choices = np.random.choice(self.h_dimension, 64)
		for i in range(len(choices)):
			plt.matshow(W[choices[i]].reshape(28, 28), cmap=plt.cm.gray)
			plt.savefig('filter%d.png' % i)
			plt.close()

