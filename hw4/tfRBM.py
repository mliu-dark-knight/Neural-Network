import os
import struct
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load MNIST data set
mnist = input_data.read_data_sets("mnist/")

class RBM(object):
	def __init__(self, v_dimension, h_dimension):
		self.v_dimension = v_dimension
		self.h_dimension = h_dimension

	def fit(self, Vs, iter=2000, batch_size=100):
		self.Vs = Vs
		self.batch_size = batch_size
		self.build(Vs, batch_size)
		self.tf_session = tf.Session()
		self.tf_session.run(tf.initialize_all_variables())
		for i in range(iter):
			batch = Vs[np.random.permutation(range(len(Vs)))[:batch_size]]
			self.train_step(batch, i)

	def build(self, Vs, batch_size):
		self.W = tf.Variable(tf.truncated_normal(shape=[self.v_dimension, self.h_dimension], stddev=0.1), name='W')
		self.b = tf.Variable(tf.truncated_normal(shape=[self.v_dimension], stddev=0.1), name='b')
		self.c = tf.Variable(tf.truncated_normal(shape=[self.h_dimension], stddev=0.1), name='c')
		self.tf_Vs = tf.placeholder(tf.float32, [None, self.v_dimension])

		self.learning_rate = tf.placeholder(tf.float32, shape=[])

		Hs = self.v_to_h(self.tf_Vs)
		E_Vs, E_Hs = self.contrastive_divergence()

		self.W_gradient_ascent = self.W.assign_add(
			self.learning_rate * tf.sub(tf.matmul(tf.transpose(self.tf_Vs), Hs), tf.matmul(tf.transpose(E_Vs), E_Hs)) / self.batch_size)
		self.b_gradient_ascent = self.b.assign_add(self.learning_rate * tf.reduce_mean(tf.sub(self.tf_Vs, E_Vs), 0) / self.batch_size)
		self.c_gradient_ascent = self.c.assign_add(self.learning_rate * tf.reduce_mean(tf.sub(Hs, E_Hs), 0) / self.batch_size)

	def train_step(self, batch, step):
		def feed():
			return {self.tf_Vs:batch, self.learning_rate: learning_rate()}

		def learning_rate():
			return 1e-1 / (1.0 + step * 1e-4)

		gradient_ascent = [self.W_gradient_ascent, self.b_gradient_ascent, self.c_gradient_ascent]
		self.tf_session.run(gradient_ascent, feed_dict=feed())

	def contrastive_divergence(self, k=10):
		V = tf.Variable(self.Vs[np.random.choice(len(self.Vs), self.batch_size)].astype(np.float32))
		for i in range(k - 1):
			H = self.v_to_h(V)
			V = self.h_to_v(H)
		H = self.v_to_h(V)
		return V, H

	def h_to_v(self, H):
		return tf.nn.sigmoid(tf.add(tf.matmul(H, tf.transpose(self.W)), self.b))

	def v_to_h(self, V):
		return tf.nn.sigmoid(tf.add(tf.matmul(V, self.W), self.c))

	def transform(self, Vs):
		return self.v_to_h(Vs.astype(np.float32)).eval(session=self.tf_session)

	def get_filters(self):
		W = self.W.eval(session=self.tf_session).T.reshape(self.h_dimension, 28, 28)
		choices = np.random.choice(self.h_dimension, 64)
		for i in range(len(choices)):
			plt.matshow(W[choices[i]], cmap=plt.cm.gray)
			plt.savefig('filter%d.png' % i)
			plt.close()


def test(train_X, train_Y, test_X, test_Y, iter=10, num='1'):
	logreg = LogisticRegression(max_iter=iter)
	logreg.fit(train_X, train_Y)

	predict_Y_train = logreg.predict(train_X)
	print("Accuracy on training data")
	print(accuracy_score(train_Y, predict_Y_train))
	predict_Y_test = logreg.predict(test_X)
	print("Accuracy on test data")
	print(accuracy_score(test_Y, predict_Y_test))
	print_CM(predict_Y_train, train_Y, predict_Y_test, test_Y, num)


def problem1(train_X, train_Y, test_X, test_Y):
	test(train_X, train_Y, test_X, test_Y, iter=10, num='1')


def problem2(train_X, train_Y, test_X, test_Y):
	rbm = RBM(v_dimension=28 * 28, h_dimension=200)
	rbm.fit(train_X)
	rbm.get_filters()
	train_X = rbm.transform(train_X)
	test_X = rbm.transform(test_X)
	test(train_X, train_Y, test_X, test_Y, num='2')


def problem3(train_X, train_Y, test_X, test_Y):
	pca = PCA(n_components=200, svd_solver='arpack')
	pca.fit(train_X)
	train_X = pca.transform(train_X)
	test_X = pca.transform(test_X)
	test(train_X, train_Y, test_X, test_Y, num='3')

def sample_bernoulli(X):
	random = np.random.rand(X.shape[0], X.shape[1])
	return np.less(random, X).astype(int)

def problem4(train_X, train_Y, test_X, test_Y):
	rbm1 = RBM(v_dimension=28 * 28, h_dimension=500)
	rbm1.fit(train_X)
	train_X = sample_bernoulli(rbm1.transform(train_X))
	rbm2 = RBM(v_dimension=500, h_dimension=200)
	rbm2.fit(train_X)
	train_X = rbm2.transform(train_X)

	test_X = rbm2.transform(sample_bernoulli(rbm1.transform(test_X)))
	test(train_X, train_Y, test_X, test_Y, num='4')


def to_binary(X):
	X[X > 0] = 1
	return X

def print_CM(train_predictions, train_labels, test_predictions, test_labels, title):
	"""Creates confusion matrix for training and test"""

	cm = confusion_matrix(train_labels, train_predictions)
	cm = cm.astype('float')

	plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Training Confusion Matrix " + title)
	plt.colorbar()
	thresh = cm.max() / 2.
	for i,j in itertools.product(range(10), range(10)):
		plt.text(j, i, round(cm[i, j], 2), horizontalalignment="center", color="white" if cm[i,j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')   	
	# plt.savefig('best_train_cm.png')
	plt.savefig('train_' + title + '.png')
	plt.close()

	# cm = confusion_matrix(targets.eval(feed_dict={y_:test_labels}), results.eval(feed_dict={x:test_data}))
	cm = confusion_matrix(test_labels, test_predictions)
	cm = cm.astype('float')
	# print(cm_normalized)

	plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Testing Confusion Matrix "+title)
	plt.colorbar()
	thresh = cm.max() / 2.
	for i,j in itertools.product(range(10), range(10)):
		plt.text(j, i, round(cm[i, j], 2), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')   	
	# plt.savefig('best_test_cm.png')
	plt.savefig('test_' + title + '.png')
	plt.close()

def main():
	train_X = mnist.train.images
	train_X = to_binary(train_X)
	train_Y = mnist.train.labels

	test_X = mnist.test.images
	test_X = to_binary(test_X)
	test_Y = mnist.test.labels


	# problem1(train_X, train_Y, test_X, test_Y)
	problem2(train_X, train_Y, test_X, test_Y)
	# problem3(train_X, train_Y, test_X, test_Y)
	# problem4(train_X, train_Y, test_X, test_Y)

main()

