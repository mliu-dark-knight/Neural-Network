import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

trainX_dir = "train/mfc/"
trainY_file = "train/lab/hw1train_labels.txt"
devX_dir = "dev/mfc/"
devY_file = "dev/lab/hw1dev_labels.txt"
testX_dir = "eval/mfc/"
testY_file = "eval/lab/hw1eval_labels.txt"
iteration = 2000
report_iteration = 20
num_labels = 2


def train(train_X, train_Y, dev_X, dev_Y, test_X, test_Y):
	assert len(train_X) == len(train_Y)
	dimension = len(train_X[0])

	x = tf.placeholder(tf.float32, [None, dimension])
	W = tf.Variable(tf.zeros([dimension, 2]))
	b = tf.Variable(tf.zeros([2]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)
	y_ = tf.placeholder(tf.float32, [None, 2])

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	learningCurve = []
	for i in range(iteration):
		batch_xs, batch_ys = train_X, train_Y
		sess.run(step, feed_dict = {x: batch_xs, y_: batch_ys})
		if i % report_iteration == 0:
			loss = 1.0 - sess.run(accuracy, feed_dict = {x: train_X, y_: train_Y})
			learningCurve.append(loss)
	plt.plot(range(0, iteration, report_iteration), learningCurve, 'b', label = "Softmax Logistic Regression")
	plt.legend(loc = 1)
	plt.show()
	
	print("train error rate: %f" % (1.0 - accuracy.eval(session=sess, feed_dict = {x: train_X, y_: train_Y})))
	print("dev error rate: %f" % (1.0 - accuracy.eval(session=sess, feed_dict = {x: dev_X, y_: dev_Y})))
	print("test error rate: %f" % (1.0 - accuracy.eval(session=sess, feed_dict = {x: test_X, y_: test_Y})))



def invalid(vector):
	return np.isnan(vector).any() or np.isinf(vector).any()


def read_XY(directory, filename):
	X, Y = {}, {}
	for file in os.listdir(directory):
		if file == ".DS_Store":
			continue
		x = np.array(list(map(float, open(directory + file, 'r').readline().split())))
		if not invalid(x):
			X[directory + file] = x

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

	return np.array(xs), np.array(ys)



def main():
	train_X, train_Y = read_XY(trainX_dir, trainY_file)
	dev_X, dev_Y = read_XY(devX_dir, devY_file)
	test_X, test_Y = read_XY(testX_dir, testY_file)
	train(train_X, train_Y, dev_X, dev_Y, test_X, test_Y)


main()
