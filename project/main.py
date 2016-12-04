import matplotlib.image as mpimg
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from DCGAN import *
from tensorflow.examples.tutorials.mnist import input_data

def mnist():
	mnist = input_data.read_data_sets("mnist/")
	real_images = mnist.train.images.reshape(-1, 28, 28, 1)
	blurred_images = np.array([blur(image, 2) for image in real_images])
	gan = DCGAN(batch_size=100, Lambda=1e1, contextual='L1')
	gan.train(real_images, blurred_images, K=10, report_iter=100, visualize_iter=100)

def CelebA():
	real_images = read_CelebA(sample_size=5500)
	# blurred_images = np.array([blur(image, 4) for image in real_images])
	masked_images = np.array([mask(image) for image in real_images])
	gan = DCGAN(image_height=218, image_width=178, image_color=3, batch_size=10, flatten_dim=14 * 12 * 32, Lambda=1e2, contextual='L1')
	# gan.train(real_images, blurred_images, report_iter=100, visualize_iter=100)
	gan.train(real_images, masked_images, report_iter=100, visualize_iter=100)

def read_CelebA(sample_size=55000):
	sample_idx = np.random.choice(202598, sample_size, replace=False)
	sample = []
	for idx in sample_idx:
		sample.append(mpimg.imread('CelebA/%0*d.jpg' % (6, idx + 1)))
	return np.array(sample)


def blur(image, std):
	return ndimage.gaussian_filter(image, sigma=(std, std, 0), order=0)

def mask(image):
	shape = image.shape
	mask = np.random.randint(2, size=(shape[0], shape[1], 1))
	return np.multiply(image, mask)


def main():
	# mnist()
	CelebA()


main()
