import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class RBM(object):
	def __init__(self, v_dimension, h_dimension):
		self.v_dimension = v_dimension #784
		self.h_dimension = h_dimension #200
		self.W = np.random.rand(self.v_dimension, self.h_dimension) * 2.0 - 1.0 #(784,200)
		self.b = np.random.rand(self.v_dimension) * 2.0 - 1.0 #(784,)
		self.c = np.random.rand(self.h_dimension) * 2.0 - 1.0 #(200,)
	
	def fit(self, Vs, iter=1000, batch_size=10):
		self.Hs = np.random.randint(2, size=(len(Vs), self.h_dimension))
		for i in range(iter):	
			sample = Vs[np.random.permutation(range(len(Vs)))[:batch_size]] #Vs = (10,784)
			dW, db, dc = self.gradient(sample)
			learning_rate = self.learning_rate(i)
			self.W += learning_rate * dW / batch_size
			self.b += learning_rate * db / batch_size
			self.c += learning_rate * dc / batch_size
		self.get_filters()

	def learning_rate(self, step):
		return 1e-1 / (1.0 + step)

	def gradient(self, Vs):
		dWs = np.zeros(self.W.shape)
		dbs = np.zeros(self.b.shape)
		dcs = np.zeros(self.c.shape)
		#Sum over all training tokens in batch
		for V in Vs:
			dW, db, dc = self.gradient_at(V)
			dWs += dW
			dbs += db
			dcs += dc
		return dWs, dbs, dcs

	def gradient_at(self, V):
		H = self.v_to_h(V)
		E_V, E_H = self.contrastive_divergence(1)
		#Implement gradient quations from CFS
		dW = np.outer(V.T, H) - np.outer(E_V.T, E_H) 
		db = V - E_V
		dc = H - E_H
		return dW, db, dc

	def contrastive_divergence(self, k):
		V = np.random.randint(2, size=(self.v_dimension))
		for i in range(k):
			H = self.v_to_h(V)
			V = self.h_to_v(H)
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
		W = self.W.T.reshape(200,28,28)
		print(W[0])
		print(W[0].shape)
		plt.matshow(W[0],cmap=plt.cm.gray)
		plt.savefig('filter.png')





