import os
import struct
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
#Load MNIST data set
mnist = input_data.read_data_sets("mnist/")

class RBM(object):
	def __init__(self, v_dimension, h_dimension):
		self.v_dimension = v_dimension
		self.h_dimension = h_dimension

	def fit(self, Vs, iter=100, batch_size=100):
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
		self.Vs = tf.placeholder(tf.float32, [None, self.v_dimension])

		self.learning_rate = tf.placeholder(tf.float32, shape=[])
		self.batch_size = tf.constant(np.float32(batch_size))

		Hs = self.v_to_h(self.Vs)
		E_Vs, E_Hs = self.contrastive_divergence()

		self.W_gradient_ascent = self.W.assign_add(
			self.learning_rate * tf.sub(tf.matmul(tf.transpose(self.Vs), Hs), tf.matmul(tf.transpose(E_Vs), E_Hs)) / self.batch_size)
		self.b_gradient_ascent = self.b.assign_add(self.learning_rate * tf.reduce_mean(tf.sub(self.Vs, E_Vs), 0) / self.batch_size)
		self.c_gradient_ascent = self.c.assign_add(self.learning_rate * tf.reduce_mean(tf.sub(Hs, E_Hs), 0) / self.batch_size)

	def train_step(self, batch, step):
		def feed():
			return {self.Vs:batch, self.learning_rate: learning_rate()}

		def learning_rate():
			return 1e-4 / (1.0 + step)

		gradient_ascent = [self.W_gradient_ascent, self.b_gradient_ascent, self.c_gradient_ascent]
		self.tf_session.run(gradient_ascent, feed_dict=feed())

	def contrastive_divergence(self, k=100):
		V = tf.constant(np.random.randint(2, size=(1, self.v_dimension)).astype(np.float32))
		for i in range(k):
			H = self.v_to_h(V)
			V = self.h_to_v(H)
		return V, H

	def h_to_v(self, H):
		return tf.nn.sigmoid(tf.add(tf.matmul(H, tf.transpose(self.W)), self.b))

	def v_to_h(self, V):
		return tf.nn.sigmoid(tf.add(tf.matmul(V, self.W), self.c))

	def transform(self, Vs):
		return self.v_to_h(Vs.astype(np.float32)).eval(session=self.tf_session)

def test(train_X, train_Y, test_X, test_Y, iter=10,num='1'):
	logreg = LogisticRegression(max_iter=iter)
	logreg.fit(train_X, train_Y)

	predict_Y_train = logreg.predict(train_X)
	print("Accuracy on training data")
	print(accuracy_score(train_Y, predict_Y_train))
	predict_Y_test = logreg.predict(test_X)
	print("Accuracy on test data")
	print(accuracy_score(test_Y, predict_Y_test))
	print_CM(predict_Y_train,train_Y,predict_Y_test,test_Y,num)


def problem1(train_X, train_Y, test_X, test_Y):
	test(train_X, train_Y, test_X, test_Y, iter=10,num='1')


def problem2(train_X, train_Y, test_X, test_Y):
	rbm = RBM(v_dimension=28 * 28, h_dimension=200)
	rbm.fit(train_X)
	train_X = rbm.transform(train_X)
	test_X = rbm.transform(test_X)
	test(train_X, train_Y, test_X, test_Y,num='2')


def problem3(train_X, train_Y, test_X, test_Y):
	pca = PCA(n_components=200, svd_solver='arpack')
	pca.fit(train_X)
	train_X = pca.transform(train_X)
	test_X = pca.transform(test_X)
	test(train_X, train_Y, test_X, test_Y,num='3')

def problem4(train_X, train_Y, test_X, test_Y):
	rbm1 = RBM(v_dimension=28 * 28, h_dimension=500)
	rbm1.fit(train_X)
	train_X = rbm1.transform(train_X)
	rbm2 = RBM(v_dimension=500, h_dimension=200)
	rbm2.fit(train_X)
	train_X = rbm2.transform(train_X)

	test_X = rbm2.transform(rbm1.transform(test_X))
	test(train_X, train_Y, test_X, test_Y,num='4')


def to_binary(X):
	X = X.reshape((-1, 28 * 28))
	X[X > 0] = 1
	return X

def print_CM(train_predictions,train_labels,test_predictions,test_labels,title):
	"""Creates confusion matrix for training and test"""

	cm = confusion_matrix(train_labels,train_predictions)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	plt.figure()
	plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
	plt.title("Training Confusion Matrix "+title)
	plt.colorbar()
	thresh = cm.max()/2.
	for i,j in itertools.product(range(10), range(10)):
		plt.text(j,i,round(cm[i,j],2),horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')   	
	# plt.savefig('best_train_cm.png')
	plt.savefig('train_'+title+'.png')
	plt.close()

	# cm = confusion_matrix(targets.eval(feed_dict={y_:test_labels}),results.eval(feed_dict={x:test_data}))
	cm = confusion_matrix(test_labels,test_predictions)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	# print(cm_normalized)

	plt.figure()
	plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
	plt.title("Testing Confusion Matrix "+title)
	plt.colorbar()
	thresh = cm.max()/2.
	for i,j in itertools.product(range(10), range(10)):
		plt.text(j,i,round(cm[i,j],2),horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')   	
	# plt.savefig('best_test_cm.png')
	plt.savefig('test_'+title+'.png')
	plt.close()

def main():
	train_X = mnist.train.images
	train_X = to_binary(train_X)
	print(train_X.shape)
	train_Y = mnist.train.labels
	print(train_Y.shape)
	test_X = mnist.test.images
	test_X = to_binary(test_X)
	test_Y = mnist.test.labels


	problem1(train_X, train_Y, test_X, test_Y)
	problem2(train_X, train_Y, test_X, test_Y)
	problem3(train_X, train_Y, test_X, test_Y)
	problem4(train_X, train_Y, test_X, test_Y)

main()

