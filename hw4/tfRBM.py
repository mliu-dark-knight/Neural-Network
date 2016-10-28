import os
import struct
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import BernoulliRBM
from sklearn.decomposition import PCA


trainX_file = "mnist/train-images.idx3-ubyte"
trainY_file = "mnist/train-labels.idx1-ubyte"
testX_file = "mnist/t10k-images.idx3-ubyte"
testY_file = "mnist/t10k-labels.idx1-ubyte"


import numpy as np
import tensorflow as tf


class RBM(UnsupervisedModel):
	def __init__(self, v_dimension, h_dimension):
		self.v_dimension = v_dimension
		self.h_dimension = h_dimension

	def fit(self, Vs, iter=10, batch_size=10):
		self.build(Vs)
		for i in range(self.num_epochs):
			batch = Vs[np.random.permutation(range(len(Vs)))[:batch_size]]
			self.train_step(batch)

	def build(self, Vs):
		self.W = tf.Variable(tf.truncated_normal(shape=[self.v_dimension, self.h_dimension], stddev=0.1), name='W')
		self.b = tf.Variable(tf.truncated_normal(shape=[self.v_dimension], stddev=0.1), name='b')
		self.c = tf.Variable(tf.truncated_normal(shape=[self.h_dimension], stddev=0.1), name='c')
		self.Vs = tf.placeholder(tf.float32, [None, self.v_dimension])


	def train_step(self, batch):
		updates = [self.w_upd8, self.bh_upd8, self.bv_upd8]
		for batch in batches:
			self.tf_session.run(updates, feed_dict=self._create_feed_dict(batch))

	def _create_feed_dict(self, data):
		return {
			self.input_data: data,
			self.hrand: np.random.rand(data.shape[0], self.num_hidden),
			self.vrand: np.random.rand(data.shape[0], data.shape[1])
		}

	def build_model(self, n_features):
		self.encode = self.sample_hidden_from_visible(self.input_data)[0]
		self.reconstruction = self.sample_visible_from_hidden(
			self.encode, n_features)

		hprob0, hstate0, vprob, hprob1, hstate1 = self.gibbs_sampling_step(
			self.input_data, n_features)
		positive = self.compute_positive_association(self.input_data,
													 hprob0, hstate0)

		nn_input = vprob

		for step in range(self.gibbs_sampling_steps - 1):
			hprob, hstate, vprob, hprob1, hstate1 = self.gibbs_sampling_step(
				nn_input, n_features)
			nn_input = vprob

		negative = tf.matmul(tf.transpose(vprob), hprob1)

		self.w_upd8 = self.W.assign_add(
			self.learning_rate * (positive - negative) / self.batch_size)

		self.bh_upd8 = self.bh_.assign_add(tf.mul(self.learning_rate, tf.reduce_mean(
			tf.sub(hprob0, hprob1), 0)))

		self.bv_upd8 = self.bv_.assign_add(tf.mul(self.learning_rate, tf.reduce_mean(
			tf.sub(self.input_data, vprob), 0)))

		vars = [self.W, self.bh_, self.bv_]
		regterm = self.compute_regularization(vars)

		self._create_cost_function_node(vprob, self.input_data,
										regterm=regterm)

	def _create_placeholders(self, n_features):

	def gibbs_sampling_step(self, visible, n_features):
		hprobs, hstates = self.sample_hidden_from_visible(visible)
		vprobs = self.sample_visible_from_hidden(hprobs, n_features)
		hprobs1, hstates1 = self.sample_hidden_from_visible(vprobs)

		return hprobs, hstates, vprobs, hprobs1, hstates1

	def sample_hidden_from_visible(self, visible):
		hprobs = tf.nn.sigmoid(tf.add(tf.matmul(visible, self.W), self.bh_))
		hstates = utilities.sample_prob(hprobs, self.hrand)

		return hprobs, hstates

	def sample_visible_from_hidden(self, hidden, n_features):
		visible_activation = tf.add(
			tf.matmul(hidden, tf.transpose(self.W)),
			self.bv_
		)

		if self.visible_unit_type == 'bin':
			vprobs = tf.nn.sigmoid(visible_activation)

		elif self.visible_unit_type == 'gauss':
			vprobs = tf.truncated_normal(
				(1, n_features), mean=visible_activation, stddev=self.stddev)

		else:
			vprobs = None

		return vprobs

	def compute_positive_association(self, visible,
									 hidden_probs, hidden_states):
		if self.visible_unit_type == 'bin':
			positive = tf.matmul(tf.transpose(visible), hidden_states)

		elif self.visible_unit_type == 'gauss':
			positive = tf.matmul(tf.transpose(visible), hidden_probs)

		else:
			positive = None

		return positive



def read(fname_img, fname_lbl):
	with open(fname_lbl, 'rb') as flbl:
		magic, num = struct.unpack(">II", flbl.read(8))
		lbl = np.fromfile(flbl, dtype=np.int8)
	flbl.close()

	with open(fname_img, 'rb') as fimg:
		magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
		img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
	fimg.close()

	return img, lbl


def test(train_X, train_Y, test_X, test_Y, iter=5):
	logreg = LogisticRegression(max_iter=iter)
	logreg.fit(train_X, train_Y)

	predict_Y = logreg.predict(train_X)
	print "Accuracy on training data"
	print accuracy_score(train_Y, predict_Y)
	predict_Y = logreg.predict(test_X)
	print "Accuracy on test data"
	print accuracy_score(test_Y, predict_Y)


def problem1(train_X, train_Y, test_X, test_Y):
	test(train_X, train_Y, test_X, test_Y, iter=10)


def problem2(train_X, train_Y, test_X, test_Y):
	rbm = BernoulliRBM(n_components=200)
	rbm.fit(train_X)
	train_X = rbm.transform(train_X)
	test_X = rbm.transform(test_X)
	test(train_X, train_Y, test_X, test_Y)


def problem3(train_X, train_Y, test_X, test_Y):
	pca = PCA(n_components=200, svd_solver='arpack')
	pca.fit(train_X)
	train_X = pca.transform(train_X)
	test_X = pca.transform(test_X)
	test(train_X, train_Y, test_X, test_Y)

def problem4(train_X, train_Y, test_X, test_Y):
	rbm1 = BernoulliRBM(n_components=500)
	rbm1.fit(train_X)
	train_X = rbm.transform(train_X)
	rbm2 = BernoulliRBM(n_components=200)
	rbm2.fit(train_X)
	train_X = rbm2.transform(train_X)

	test_X = rbm2.transform(rbm1.transform(test_X))
	test(train_X, train_Y, test_X, test_Y)


def to_binary(X):
	X = X.reshape((-1, 28 * 28))
	X[X > 0] = 1
	return X



def main():
	train_X, train_Y = read(trainX_file, trainY_file)
	train_X = to_binary(train_X)
	test_X, test_Y = read(testX_file, testY_file)
	test_X = to_binary(test_X)

	problem1(train_X, train_Y, test_X, test_Y)
	problem2(train_X, train_Y, test_X, test_Y)
	problem3(train_X, train_Y, test_X, test_Y)
	problem4(train_X, train_Y, test_X, test_Y)

main()

