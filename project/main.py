import matplotlib.image as mpimg
import numpy as np
from DCGAN import *
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("mnist/", one_hot=True)



def main():
	data = read_CelebA(sample_size=550)
	gan = DCGAN(image_height=218, image_width=178, image_color=3, batch_size=100, flatten_dim=14 * 12 * 32)
	gan.train(data, data)
	# gan.train(mnist.train.images.reshape(-1, 28, 28, 1), mnist.train.images.reshape(-1, 28, 28, 1))


def read_CelebA(sample_size=55000):
	sample_idx = np.random.choice(202598, sample_size, replace=False)
	sample = []
	for idx in sample_idx:
		sample.append(mpimg.imread('CelebA/%0*d.jpg' % (6, idx)))
	return np.array(sample)


main()
