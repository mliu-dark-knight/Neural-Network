import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist/", one_hot=True)



class RNN(object):
    def __init__(self, lstm=False, n_unit=100, n_step=28, n_input=28, n_class=10):
        self.n_unit = n_unit
        self.n_step = n_step
        self.n_input = n_input
        self.n_class = n_class
        self.lstm = lstm

    def fit(self, iterations=1000, report_iter=100, batch_size=100):
        def learning_rate(step) :
            return 1e-1 / (1.0 + step * 1e-4)

        self.build()
        self.tf_session = tf.Session()
        self.tf_session.run(tf.initialize_all_variables())
        learning_curve = []

        train_X, train_Y = mnist.train.images.reshape(-1, self.n_step, self.n_input), mnist.train.labels

        for i in range(iterations):
            batch_idx = np.random.choice(len(train_X), batch_size)
            batch_x, batch_y = train_X[batch_idx], train_Y[batch_idx]
            self.tf_session.run(self.step, feed_dict={self.X: batch_x, self.Y: batch_y, self.learning_rate: learning_rate(i)})

            if i % report_iter == 0:
                accuracy = self.eval(train_X, train_Y)
                learning_curve.append(accuracy)
                print("Training accuracy: %f" % accuracy)
        return learning_curve


    def build(self):
        if self.lstm:
            self.cell = rnn_cell.BasicLSTMCell(self.n_unit, state_is_tuple=True)
        else:

            # self.cell = rnn_cell.BasicRNNCell(self.n_unit, state_is_tuple=True)
            self.cell = rnn_cell.BasicRNNCell(self.n_unit)
        self.X = tf.placeholder(tf.float32, [None, self.n_step, self.n_input])
        self.Y = tf.placeholder(tf.float32, [None, self.n_class])
        self.W = tf.Variable(tf.truncated_normal([self.n_unit, self.n_class], stddev=0.1))
        self.b = tf.Variable(tf.constant(0.1, shape=[self.n_class]))

        rnn_input = tf.split(0, self.n_step, tf.reshape(tf.transpose(self.X, [1, 0, 2]), [-1, self.n_input]))
        output, state = rnn.rnn(self.cell, rnn_input, dtype=tf.float32)

        prediction = tf.matmul(output[-1], self.W) + self.b
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(prediction, self.Y)

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy)

    def eval(self, X, Y):
        return self.tf_session.run(self.accuracy, feed_dict={self.X: X, self.Y: Y})

    def test(self):
        test_X, test_Y = mnist.test.images.reshape(-1, self.n_step, self.n_input), mnist.test.labels
        print("Testing accuracy: %f" % self.eval(test_X, test_Y))


def plot_convergence(curve,filename):
    plt.plot(curve)
    plt.title(filename+' Accuracy Convergence')
    plt.xlabel('Training Iteration')
    plt.ylabel('Training Corpus Accuracy')
    plt.savefig(filename+'.png')  
    plt.close()

def main():
    # rnn = RNN(lstm=False, n_step=784, n_input=1)
    # curve = rnn.fit()
    # rnn.test()

    rnn = RNN(lstm=False, n_step=28, n_input=28)
    rnn.fit()
    rnn.test()


    # rnn = RNN(lstm=True, n_step=784, n_input=1)
    # rnn.fit()
    # rnn.test()

    # rnn = RNN(lstm=True, n_step=28, n_input=28)
    # curve = rnn.fit()
    # rnn.test()
    # plot_convergence(curve,'lstm_28_28')


main()
