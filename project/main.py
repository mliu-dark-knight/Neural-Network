import matplotlib.image as mpimg
import numpy as np
import scipy.ndimage as ndimage
from DCGAN import *
from tensorflow.examples.tutorials.mnist import input_data

def mnist():
	mnist = input_data.read_data_sets("mnist/")
	real_images = mnist.train.images.reshape(-1, 28, 28, 1)
	blurred_images = np.array([blur(image) for image in real_images])
	gan = DCGAN(Lambda=1e1, contextual='L1')
	gan.train(real_images, blurred_images, K=10, report_iter=100, visualize_iter=100)

def CelebA():
	data = read_CelebA(sample_size=5500)
	gan = DCGAN(image_height=218, image_width=178, image_color=3, batch_size=100, flatten_dim=14 * 12 * 32, contextual='L2')
	gan.train(data, data, report_iter=10, visualize_iter=10)

def read_CelebA(sample_size=55000):
	sample_idx = np.random.choice(202598, sample_size, replace=False)
	sample = []
	for idx in sample_idx:
		sample.append(mpimg.imread('CelebA/%0*d.jpg' % (6, idx)))
	return np.array(sample)


def blur(image):
	return ndimage.gaussian_filter(image, sigma=(2, 2, 0), order=0)


def main():
	mnist()
	# CelebA()


main()
