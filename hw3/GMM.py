import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import matrix_rank

class GMM(object):
	def __init__(self, m):
		self.m = m

	def init_params(self, X):
		split = np.array_split(X, self.m)
		self.w = np.array([len(s) / float(len(X)) for s in split])
		self.mu = np.array([np.mean(s, axis=0) for s in split])
		self.sigma = np.array([np.cov(s.T) for s in split])

	def fit(self, X, iteration):
		self.init_params(X)
		for i in xrange(iteration):
			self.E_step(X)
			self.M_step(X)
		print "w:"
		print self.w
		print "mu:"
		print self.mu
		print "sigma:"
		print self.sigma

	def E_step(self, X):
		self.gamma = self.inference(X)

	def M_step(self, X):
		sum_gamma = np.sum(self.gamma, axis=1)
		self.w = sum_gamma / len(X)
		self.mu = np.dot(self.gamma, X) / sum_gamma.reshape((-1, 1))
		# print self.mu
		outer = np.array([np.tensordot(self.gamma[k], np.array([np.outer(x - self.mu[k], x - self.mu[k]) for x in X]), axes=([0], [0])) for k in xrange(self.m)])
		self.sigma = outer / sum_gamma.reshape((-1, 1, 1))

		for k in xrange(self.m):
			if matrix_rank(self.sigma[k]) < len(X[0]):
				noise = np.zeros((len(X[0]), len(X[0])))
				np.fill_diagonal(noise, 1e-4)
				self.sigma[k] += noise

	def inference(self, X):
		try:
			normal = np.array([self.w[k] * multivariate_normal.pdf(X, mean=self.mu[k], cov=self.sigma[k]) for k in xrange(self.m)])
		except:
			print self.sigma
		return normal / np.sum(normal, axis=0)

	def clustering(self, X):
		inf = self.inference(X)
		return np.array([self.mu[np.argmax(inf[:, i])] for i in xrange(len(X))])

