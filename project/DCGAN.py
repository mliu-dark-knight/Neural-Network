import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class DCGAN(object):
	def __init__(self, image_height=28, image_width=28, image_color=1, batch_size=100, 
				 g_kernel_size=4, g_channel_1=4, g_channel_2=8, g_channel_3=4,
				 d_kernel_size=4, d_channel_1=4, d_channel_2=8, d_channel_3=16, d_channel_4=32,
				 flatten_dim=128, hidden_dim=64, Lambda=1e1):

		self.batch_size = batch_size
		self.image_height = image_height
		self.image_width = image_width
		self.image_color = image_color
		self.flatten_dim = flatten_dim
		self.hidden_dim = hidden_dim
		self.Lambda = Lambda

		self.g_kernel_size = g_kernel_size
		self.d_kernel_size = d_kernel_size

		self.g_channel_1 = g_channel_1
		self.g_channel_2 = g_channel_2
		self.g_channel_3 = g_channel_3
		self.d_channel_1 = d_channel_1
		self.d_channel_2 = d_channel_2
		self.d_channel_3 = d_channel_3
		self.d_channel_4 = d_channel_4

		self.build_model()

	def weight_variable(self, shape, name=None, reuse=False):
		if reuse:
			with tf.variable_scope('', reuse=True):
				return tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=1e-1))
		return tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=1e-1))

	def bias_variable(self, shape, name=None, reuse=False):
		if reuse:
			with tf.variable_scope('', reuse=True):
				return tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(mean=1e-1, stddev=1e-1))
		return tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(mean=1e-1, stddev=1e-1))


	def build_model(self):
		self.real_images = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_color], name='real_images')
		self.blurred_images = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_color], name='blurred_images')
		self.generated_images = self.generator(self.blurred_images)

		self.D_real = self.discriminator(self.real_images)
		self.D_generated = self.discriminator(self.generated_images, reuse=True)

		self.d_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.D_real, tf.one_hot(indices=np.ones(self.batch_size).astype(int), depth=2, on_value=1.0, off_value=0.0)))
		self.d_loss_generated = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.D_generated, tf.one_hot(indices=np.zeros(self.batch_size).astype(int), depth=2, on_value=1.0, off_value=0.0)))
		self.d_loss = self.d_loss_real + self.d_loss_generated

		self.g_loss_perceptual = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.D_generated, tf.one_hot(indices=np.ones(self.batch_size).astype(int), depth=2, on_value=1.0, off_value=0.0)))
		self.g_loss_contextual = tf.reduce_sum(tf.contrib.layers.flatten(tf.abs(self.generated_images - self.real_images))) / (self.image_height * self.image_width * self.image_color)
		self.g_loss = self.g_loss_contextual + self.Lambda * self.g_loss_perceptual

		self.d_variables = [variable for variable in tf.trainable_variables() if variable.name.startswith('d_')]
		self.g_variables = [variable for variable in tf.trainable_variables() if variable.name.startswith('g_')]

		self.learning_rate = tf.placeholder(tf.float32, shape=[])

		self.d_gradient_descent = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.d_loss, var_list=self.d_variables)
		self.g_gradient_descent = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.g_loss, var_list=self.g_variables)

		def check_scope():
			for variable in tf.trainable_variables():
				print(variable.name, variable.get_shape())

		# check_scope()

	def train(self, real_images, blurred_images, K=1, iteration=1000, report_iter=100, visualize_iter=100):
		def generator_learning_rate(step):
			return 1e-4 / (1 + step * 1e-4)

		def discriminator_learning_rate(step):
			return 1e-2 / (1 + step * 1e-4)

		self.tf_session = tf.Session()
		self.tf_session.run(tf.initialize_all_variables())

		for i in range(iteration):
			batch_idx = np.random.choice(len(real_images), self.batch_size, replace=False)
			batch_real_images, batch_blurred_images = real_images[batch_idx], blurred_images[batch_idx]
			for k in range(K):
				self.tf_session.run(self.d_gradient_descent, feed_dict={self.real_images: batch_real_images, self.blurred_images: batch_blurred_images, self.learning_rate: discriminator_learning_rate(i)})

			# discriminator loss before updating generator
			if i % report_iter == 0:
				d_loss = self.tf_session.run(self.d_loss, feed_dict={self.real_images: batch_real_images, self.blurred_images: batch_blurred_images})
				print('discriminator loss: %f' % d_loss)

			self.tf_session.run(self.g_gradient_descent, feed_dict={self.real_images: batch_real_images, self.blurred_images: batch_blurred_images, self.learning_rate: generator_learning_rate(i)})

			if i % report_iter == 0:
				d_loss = self.tf_session.run(self.d_loss, feed_dict={self.real_images: batch_real_images, self.blurred_images: batch_blurred_images})
				print('discriminator loss: %f' % d_loss)
				g_loss_contextual = self.tf_session.run(self.g_loss_contextual, feed_dict={self.real_images: batch_real_images, self.blurred_images: batch_blurred_images})
				print('generator contextual loss: %f' % (g_loss_contextual / self.batch_size))

			if i % visualize_iter == 0:
				self.show_generated_image(blurred_images[np.random.randint(len(blurred_images), size=1)])
		self.show_generated_image(blurred_images[np.random.randint(len(blurred_images), size=1)])

	def show_generated_image(self, blurred_images):
		generated_images = self.tf_session.run(self.generated_images, feed_dict={self.blurred_images: blurred_images})
		for generated_image in generated_images:
			if self.image_color == 1:
				plt.matshow(np.squeeze(generated_image), cmap=plt.cm.gray)
			else:
				plt.imshow(generated_image)
			plt.show()

	def print_variables(self, names=None):
		for name in names:
			print(name)
			print([self.tf_session.run(variable) for variable in tf.trainable_variables() if variable.name.startswith(name)][0])


	def discriminator(self, images, reuse=False):
		W_1 = self.weight_variable([self.d_kernel_size, self.d_kernel_size, self.image_color, self.d_channel_1], name='d_w1', reuse=reuse)
		b_1 = self.bias_variable([self.d_channel_1], name='d_b1', reuse=reuse)
		h_1 = tf.nn.relu(tf.nn.conv2d(images, W_1, strides=[1, 2, 2, 1], padding='SAME') + b_1)

		W_2 = self.weight_variable([self.d_kernel_size, self.d_kernel_size, self.d_channel_1, self.d_channel_2], name='d_w2', reuse=reuse)
		b_2 = self.bias_variable([self.d_channel_2], name='d_b2', reuse=reuse)
		h_2 = tf.nn.relu(tf.nn.conv2d(h_1, W_2, strides=[1, 2, 2, 1], padding='SAME') + b_2)

		W_3 = self.weight_variable([self.d_kernel_size, self.d_kernel_size, self.d_channel_2, self.d_channel_3], name='d_w3', reuse=reuse)
		b_3 = self.bias_variable([self.d_channel_3], name='d_b3', reuse=reuse)
		h_3 = tf.nn.relu(tf.nn.conv2d(h_2, W_3, strides=[1, 2, 2, 1], padding='SAME') + b_3)

		W_4 = self.weight_variable([self.d_kernel_size, self.d_kernel_size, self.d_channel_3, self.d_channel_4], name='d_w4', reuse=reuse)
		b_4 = self.bias_variable([self.d_channel_4], name='d_b4', reuse=reuse)
		h_4 = tf.nn.relu(tf.nn.conv2d(h_3, W_4, strides=[1, 2, 2, 1], padding='SAME') + b_4)

		h_4 = tf.reshape(h_4, [self.batch_size, self.flatten_dim])
		W_5 = self.weight_variable([self.flatten_dim, self.hidden_dim], name='d_w5', reuse=reuse)
		d_5 = self.bias_variable([self.hidden_dim], name='d_b5', reuse=reuse)
		h_5 = tf.nn.sigmoid(tf.matmul(h_4, W_5) + d_5)

		W_6 = self.weight_variable([self.hidden_dim, 2], name='d_w6', reuse=reuse)
		d_6 = self.bias_variable([2], name='d_b6', reuse=reuse)
		h_6 = tf.matmul(h_5, W_6) + d_6

		return h_6


	def generator(self, images):
		W_1 = self.weight_variable([self.g_kernel_size, self.g_kernel_size, self.image_color, self.g_channel_1], name='g_w1')
		b_1 = self.bias_variable([self.g_channel_1], name='g_b1')
		h_1 = tf.nn.relu(tf.nn.conv2d(images, W_1, strides=[1, 1, 1 ,1], padding='SAME') + b_1)

		W_2 = self.weight_variable([self.g_kernel_size, self.g_kernel_size, self.g_channel_1, self.g_channel_2], name='g_w2')
		b_2 = self.bias_variable([self.g_channel_2], name='g_b2')
		h_2 = tf.nn.relu(tf.nn.conv2d(h_1, W_2, strides=[1, 1, 1 ,1], padding='SAME') + b_2)

		W_3 = self.weight_variable([self.g_kernel_size, self.g_kernel_size, self.g_channel_2, self.g_channel_3], name='g_w3')
		b_3 = self.bias_variable([self.g_channel_3], name='g_b3')
		h_3 = tf.nn.relu(tf.nn.conv2d(h_2, W_3, strides=[1, 1, 1 ,1], padding='SAME') + b_3)

		W_4 = self.weight_variable([self.g_kernel_size, self.g_kernel_size, self.g_channel_1, self.image_color], name='g_w4')
		b_4 = self.bias_variable([self.image_color], name='g_b4')
		h_4 = tf.nn.relu(tf.nn.conv2d(h_1, W_4, strides=[1, 1, 1 ,1], padding='SAME') + b_4)	

		return h_4

