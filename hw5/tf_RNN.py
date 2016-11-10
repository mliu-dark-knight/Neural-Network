from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BATCH_SIZE = 100

#Load MNIST data set
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#Training images (55000,784)
# X = mnist.train.images
# print("Input Data shape:",X.shape)

#Create an interactive TF session
sess = tf.InteractiveSession()

#Define model
x = tf.placeholder(tf.float32,[None,784])
W_hid = tf.Variable(tf.zeros([1,100]))
W_out = tf.Variable(tf.zeros([100, 10]))
b_hid = tf.Variable(tf.zeros([100]))
b_out = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder(tf.float32,[None,10])

#Reshape input batch for RNN 
X = tf.reshape(x,[-1,784,1])
X = tf.transpose(X,[1,0,2]) # swap batch and steps
X =tf.reshape(X,[-1,1])
X = tf.matmul(X,W_hid)+b_hid
X = tf.split(0,784,x)

cell = tf.nn.rnn_cell.BasicLSTMCell(100,forget_bias=1.0)
init_state = tf.zeros([BATCH_SIZE, cell.state_size])

outputs,states = tf.nn.rnn(cell,X,initial_state=init_state)
y = tf.matmul(outputs[-1],W_out) + b_out


#Define error function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#Define training step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
	print(i)
	batch_xs,batch_ys = mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})


#Define model evaluation 
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

