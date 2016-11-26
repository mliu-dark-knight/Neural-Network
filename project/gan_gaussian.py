import numpy as np
from scipy.stats import norm
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Constants
MU = 1			
SIGMA = 1
RANGE = 5.0
SCALE = 0.01
LEARNING_RATE = 0.01

#Training parameters
NUM_ITERS = 100000	#Number of iterations for training
K = 1 				#Number of steps to apply to discriminator 
M = 10				#Minibatch size


def sampleData(n):
	"""
	Sample the data distribution.
	"""
	return np.sort(np.random.normal(MU,SIGMA,n))


def sampleGen(n):
	"""
	Sample the generator (noise) distribution using stratified sampling.
	"""
	perturb = np.random.random(n)
	return np.linspace(-RANGE,RANGE,n) + perturb*SCALE


class GAN():
	def __init__(self):
		pass


	def weight_variable(self, shape, name=None, reuse=False):
		if reuse:
			with tf.variable_scope('', reuse=True):
				return tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=1e-1))
		return tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=1e-1))

	def bias_variable(self, shape, name=None, reuse=False):
		if reuse:
			with tf.variable_scope('', reuse=True):
				return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0.0))
		return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0.0))

	def build(self):
		#Weights/Biases for Discriminator Network
		# self.w1_d = tf.Variable(tf.random_normal([M,6]))
		# self.b1_d = tf.constant(0.0, shape=[6])
		# self.w2_d = tf.Variable(tf.random_normal([6,5]))
		# self.b2_d = tf.constant(0.0, shape=[5])
		# self.w3_d = tf.Variable(tf.random_normal([5,1]))
		# self.b3_d = tf.constant(0.0, shape=[1])

		#Weights/Biases for Generator Network
		# self.w1_g = tf.Variable(tf.random_normal([M,6]))
		# self.b1_g = tf.constant(0.0, shape=[6])
		# self.w2_g = tf.Variable(tf.random_normal([6,5]))
		# self.b2_g = tf.constant(0.0, shape=[5])
		# self.w3_g = tf.Variable(tf.random_normal([5,1]))
		# self.b3_g = tf.constant(0.0, shape=[1])

		#Inputs for Discriminator and Generator
		self.input_d = tf.placeholder(tf.float32,shape=[M,1])
		# self.d_labels = tf.placeholder(tf.float32)
		self.input_g = tf.placeholder(tf.float32,shape=[M,1])
		# self.g_lables = 

		self.G = self.generator(self.input_g)
		self.D = self.discriminator(self.input_d)
		self.D1 = tf.maximum(tf.minimum(self.D,0.99),0.01) #clamp to probability
		self.D2 = tf.maximum(tf.minimum(self.G,0.99),0.01)


		#Feedforward for Generator
		# g_1 = tf.nn.tanh(tf.matmul(tf.transpose(self.g_input),self.w1_g)+self.b1_g)
		# g_2 = tf.nn.tanh(tf.matmul(g_1,self.w2_g)+self.b2_g)
		# G_out = tf.nn.tanh(tf.matmul(g_2,self.w3_g)+self.b3_g)
		# self.G = tf.mul(RANGE,G_out)

		#Feedforward for Discriminator
		# d_1 = tf.nn.tanh(tf.matmul(tf.transpose(self.d_input),self.w1_d)+self.b1_d)
		# d_2 = tf.nn.tanh(tf.matmul(d_1,self.w2_d)+self.b2_d)
		# D_out = tf.nn.tanh(tf.matmul(d_2,self.w3_d)+self.b3_d)
		# self.D1 = tf.maximum(tf.minimum(D_out,0.99),0.01) #clamp to probability
		# self.D2 = tf.maximum(tf.minimum(self.G,0.99),0.01)

		#Loss Functions
		self.d_loss = tf.reduce_mean(tf.log(self.D1)+tf.log(1-self.D2))
		self.g_loss = tf.reduce_mean(tf.log(self.D2))


		self.d_variables = [variable for variable in tf.trainable_variables() if variable.name.startswith('d_')]
		self.g_variables = [variable for variable in tf.trainable_variables() if variable.name.startswith('g_')]
		#Optimizers
		self.optimizer_d = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(1-self.d_loss,var_list=self.d_variables)
		self.optimizer_g = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(1-self.g_loss,var_list=self.g_variables)

	def discriminator(self,d_input,reuse=False):
		W_1 = self.weight_variable([M,6],name='d_w1',reuse=reuse)
		b_1 = self.bias_variable([6],name='d_b1',reuse=reuse)
		h_1 = tf.nn.tanh(tf.matmul(tf.transpose(d_input),W_1)+b_1)

		W_2 = self.weight_variable([6,5],name='d_w2',reuse=reuse)
		b_2 = self.bias_variable([5],name='d_b2',reuse=reuse)
		h_2 = tf.nn.tanh(tf.matmul(h_1,W_2)+b_2)

		W_3 = self.weight_variable([5,1],name='d_w3',reuse=reuse)
		b_3 = self.bias_variable([1],name='d_b3',reuse=reuse)
		h_3 = tf.nn.tanh(tf.matmul(h_2,W_3)+b_3)

		return tf.mul(RANGE,h_3)

	def generator(self,g_input):
		W_1 = self.weight_variable([M,6],name='g_w1')
		b_1 = self.bias_variable([6],name='g_b1')
		h_1 = tf.nn.tanh(tf.matmul(tf.transpose(g_input),W_1)+b_1)

		W_2 = self.weight_variable([6,5],name='g_w2')
		b_2 = self.bias_variable([5],name='g_b2')
		h_2 = tf.nn.tanh(tf.matmul(h_1,W_2)+b_2)

		W_3 = self.weight_variable([5,1],name='g_w3')
		b_3 = self.bias_variable([1],name='g_b3')
		h_3 = tf.nn.tanh(tf.matmul(h_2,W_3)+b_3)	

		return h_3	



	def train(self):
		self.build()
		self.sess = tf.Session()
		self.sess.run(tf.initialize_all_variables())
		for i in range(NUM_ITERS):
			print(i)
			for k in range(K):
				z = sampleGen(M) 
				x = sampleData(M)
				self.sess.run(self.optimizer_d,{self.input_d:np.reshape(x,(M,1)),self.input_g:np.reshape(z,(M,1))})
			z = sampleGen(M)
			self.sess.run(self.optimizer_g,{self.input_g:np.reshape(z,(M,1))})
			d_loss = self.sess.run(self.d_loss, feed_dict={self.})
			print('discriminator loss: %f' % d_loss)


	def plot_fig(self):
	    # plots pg, pdata, decision boundary 
	    f,ax=plt.subplots(1)
	    # p_data
	    xs=np.linspace(-5,5,1000)
	    ax.plot(xs, norm.pdf(xs,loc=MU,scale=SIGMA), label='p_data')

	    # decision boundary
	    r=5000 # resolution (number of points)
	    xs=np.linspace(-5,5,r)
	    ds=np.zeros((r,1)) # decision surface
	    # process multiple points in parallel in same minibatch
	    for i in range(int(r/M)):
	        x=np.reshape(xs[M*i:M*(i+1)],(M,1))
	        ds[M*i:M*(i+1)]=self.sess.run(self.D1,{self.input_d: x})

	    ax.plot(xs, ds, label='decision boundary')

	    # distribution of inverse-mapped points
	    zs=np.linspace(-5,5,r)
	    gs=np.zeros((r,1)) # generator function
	    for i in range(int(r/M)):
	        z=np.reshape(zs[M*i:M*(i+1)],(M,1))
	        gs[M*i:M*(i+1)]=self.sess.run(self.G,{self.input_g: z})
	    histc, edges = np.histogram(gs, bins = 10)
	    ax.plot(np.linspace(-5,5,10), histc/float(r), label='p_g')

	    # ylim, legend
	    ax.set_ylim(0,1.1)
	    plt.legend()
	    plt.savefig('gan.png')

	def test(self):
		self.plot_fig()


def main():
	gan = GAN()
	gan.train()
	gan.test()

main()