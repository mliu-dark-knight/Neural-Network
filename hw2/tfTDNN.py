import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix


trainX_dir = "train/fbc/"
trainY_file = "train/lab/hw2train_labels.txt"
testX_dir = "eval/fbc/"
testY_file = "eval/lab/hw2eval_labels.txt"
iteration = 100000
report_iteration = 10000
frame_limit = 70
batch_size = 10
num_labels = 9


def train(train_X, train_Y, test_X, test_Y, iteration, batch_size):
	assert len(train_X) == len(train_Y)
	time, frequency = len(train_X[0][0]), len(train_X[0])
	assert frequency == 16 and time == 70

	x_image = tf.placeholder(tf.float32, [None, frequency, time])

	x_conv1 = tf.reshape(x_image, [-1, 16, 70, 1])
	W_conv1 = weight_variable([16, 3, 1, 8])
	b_conv1 = bias_variable([8])
	h_conv1 = tf.nn.relu(conv2d(x_conv1, W_conv1) + b_conv1)

	x_conv2 = tf.transpose(h_conv1, perm=[0, 3, 2, 1])
	W_conv2 = weight_variable([8, 5, 1, 3])
	b_conv2 = bias_variable([3])
	h_conv2 = tf.nn.relu(conv2d(x_conv2, W_conv2) + b_conv2)

	x_conv3 = tf.transpose(h_conv2, perm=[0, 3, 2, 1])
	W_conv3 = weight_variable([3, 2, 1, 1])
	b_conv3 = bias_variable([1])
	h_conv3 = tf.nn.relu(conv2d(x_conv3, W_conv3) + b_conv3)

	x = tf.reshape(h_conv3, [-1, 63])
	W = tf.Variable(tf.zeros([63, num_labels]))
	b = tf.Variable(tf.zeros([num_labels]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)
	y_ = tf.placeholder(tf.float32, [None, num_labels])

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	l_rate = tf.placeholder(tf.float32, shape=[])
	step = tf.train.GradientDescentOptimizer(l_rate).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	for i in range(iteration):
		batch_idx = np.random.permutation(range(len(train_X)))[:batch_size]
		batch_xs, batch_ys = train_X[batch_idx], train_Y[batch_idx]
		sess.run(step, feed_dict = {x_image: batch_xs, y_: batch_ys, l_rate: learning_rate(i)})
		if i % report_iteration == 0:
			loss = 1.0 - sess.run(accuracy, feed_dict = {x_image: train_X, y_: train_Y})
			print(loss)

	train_Y = np.array([np.argmax(train_y) for train_y in train_Y])
	test_Y = np.array([np.argmax(test_y) for test_y in test_Y])
	predict_Y = sess.run(tf.argmax(y, 1), feed_dict={x_image: train_X})
	print("Accuracy on training data")
	print(accuracy_score(train_Y, predict_Y))
	print(confusion_matrix(train_Y, predict_Y))
	predict_Y = sess.run(tf.argmax(y, 1), feed_dict={x_image: test_X})
	print("Accuracy on test data")
	print(accuracy_score(test_Y, predict_Y))
	print(confusion_matrix(test_Y, predict_Y))


def train_trial(train_X, train_Y, test_X, test_Y, iteration, batch_size):
	assert len(train_X) == len(train_Y)
	time, frequency = len(train_X[0][0]), len(train_X[0])
	assert frequency == 16 and time == 70

	x_image = tf.placeholder(tf.float32, [None, frequency, time])

	x_conv1 = tf.reshape(x_image, [-1, 16, 70, 1])
	W_conv1 = weight_variable([16, 3, 1, 8])
	b_conv1 = bias_variable([8])
	h_conv1 = tf.nn.relu(conv2d(x_conv1, W_conv1) + b_conv1)

	x_conv2 = tf.transpose(h_conv1, perm=[0, 3, 2, 1])
	W_conv2 = weight_variable([8, 5, 1, 3])
	b_conv2 = bias_variable([3])
	h_conv2 = tf.nn.relu(conv2d(x_conv2, W_conv2) + b_conv2)

	x_conv3 = tf.transpose(h_conv2, perm=[0, 3, 2, 1])
	W_conv3 = weight_variable([3, 2, 1, 2])
	b_conv3 = bias_variable([2])
	h_conv3 = tf.nn.relu(conv2d(x_conv3, W_conv3) + b_conv3)

	keep_prob = tf.placeholder(tf.float32)
	h_drop = tf.nn.dropout(h_conv3, keep_prob)

	x = tf.reshape(h_drop, [-1, 63 * 2])
	W = tf.Variable(tf.zeros([63 * 2, num_labels]))
	b = tf.Variable(tf.zeros([num_labels]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)
	y_ = tf.placeholder(tf.float32, [None, num_labels])

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	l_rate = tf.placeholder(tf.float32, shape=[])
	step = tf.train.GradientDescentOptimizer(l_rate).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	for i in range(iteration):
		batch_idx = np.random.permutation(range(len(train_X)))[:batch_size]
		batch_xs, batch_ys = train_X[batch_idx], train_Y[batch_idx]
		sess.run(step, feed_dict = {x_image: batch_xs, y_: batch_ys, keep_prob: 0.5, l_rate: learning_rate(i)})
		if i % report_iteration == 0:
			loss = 1.0 - sess.run(accuracy, feed_dict = {x_image: train_X, y_: train_Y, keep_prob: 1.0})
			print(loss)

	train_Y = np.array([np.argmax(train_y) for train_y in train_Y])
	test_Y = np.array([np.argmax(test_y) for test_y in test_Y])
	predict_Y = sess.run(tf.argmax(y, 1), feed_dict={x_image: train_X, keep_prob: 1.0})
	print("Accuracy on training data")
	print(accuracy_score(train_Y, predict_Y))
	print(confusion_matrix(train_Y, predict_Y))
	predict_Y = sess.run(tf.argmax(y, 1), feed_dict={x_image: test_X, keep_prob: 1.0})
	print("Accuracy on test data")
	print(accuracy_score(test_Y, predict_Y))
	print(confusion_matrix(test_Y, predict_Y))


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def learning_rate(step):
	return 1e-2 / (1 + step * 1e-4)



def invalid(vector):
	return np.isnan(vector).any() or np.isinf(vector).any()

def read_single_X(file):
	X = np.empty((frame_limit, 16))
	count = 0
	with open(file, 'r') as f:
		for line in f:
			x = np.array(list(map(float, line.split())))
			if invalid(x):
				raise ValueError()
			X[count] = x
			count += 1
			if count == frame_limit:
				break
	f.close()
	if count != frame_limit:
		raise ValueError()
	return X

def read_XY(directory, filename):
	X, Y = {}, {}
	for file in os.listdir(directory):
		if file == ".DS_Store":
			continue
		try:
			x = read_single_X(directory + file)
		except:
			pass
		else:
			X[directory + file] = x.T

	with open(filename, 'r') as f:
		for line in f:
			line = line[:-1].split()
			y = np.zeros(num_labels)
			y[int(line[0])] = 1
			Y[line[1]] = y

	xs, ys = [], []
	for file in X:
		if file in Y:
			xs.append(X[file])
			ys.append(Y[file])

	return np.array(xs).astype(np.float64), np.array(ys).astype(np.float64)


def main():
	train_X, train_Y = read_XY(trainX_dir, trainY_file)
	test_X, test_Y = read_XY(testX_dir, testY_file)
	train(train_X, train_Y, test_X, test_Y, iteration, batch_size)
	train_trial(train_X, train_Y, test_X, test_Y, iteration, batch_size)

main()
