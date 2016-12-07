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
	gan.train(real_images, blurred_images, report_iter=100, visualize_iter=100)
	sample_idx = np.random.randint(len(blurred_images), size=10)
	generated_images = gan.reconstruct_image(blurred_images[sample_idx])
	save_grayscale(real_images[sample_idx], blurred_images[sample_idx], generated_images, 'result/mnist')


def CelebA():
	real_images = read_CelebA(sample_size=10000)

	blurred_images = np.array([blur(image, 4) for image in real_images])
	blur_gan = DCGAN(image_height=218, image_width=178, image_color=3, batch_size=10, g_kernel_size=8, d_kernel_size=8, flatten_dim=14 * 12 * 32, hidden_dim=256, Lambda=1e1, contextual='L1')
	blur_gan.train(real_images, blurred_images, report_iter=10, visualize_iter=10)

	masked_images = np.array([mask(image) for image in real_images])
	mask_gan = DCGAN(image_height=218, image_width=178, image_color=3, batch_size=10, flatten_dim=14 * 12 * 32, Lambda=1e2, contextual='L1')
	mask_gan.train(real_images, masked_images, iteration=10000, report_iter=1000, visualize_iter=10000)
	sample_idx = np.random.randint(len(masked_images), size=10)
	generated_images = mask_gan.reconstruct_image(masked_images[sample_idx])
	save_images(real_images[sample_idx], masked_images[sample_idx].astype(np.uint8), generated_images.astype(np.uint8), 'result/CelebA', 'masked')

	down_sampled_images = np.array([down_sample(image, 2) for image in real_images])
	down_sample_gan = DCGAN(image_height=218, image_width=178, image_color=3, batch_size=10, stddev=1e-2, g_kernel_size=8, flatten_dim=14 * 12 * 32, hidden_dim=256, Lambda=1e-1, contextual='L1')
	down_sample_gan.train(real_images, down_sampled_images, alpha=1e-2, report_iter=100, visualize_iter=1000)


def save_images(real_images, blurred_images, generated_images, prefix, type):
	for i in range(len(real_images)):
		plt.imshow(real_images[i])
		plt.axis('off')
		plt.savefig(prefix + '_real_' + str(i) + '.png', bbox_inches='tight')
		plt.imshow(blurred_images[i])
		plt.axis('off')
		plt.savefig(prefix + '_' + type + '_' + str(i) + '.png', bbox_inches='tight')
		plt.imshow(generated_images[i])
		plt.axis('off')
		plt.savefig(prefix + '_generated_' + str(i) + '.png', bbox_inches='tight')


def save_grayscale(real_images, blurred_images, generated_images, prefix):
	for i in range(len(real_images)):
		plt.matshow(np.squeeze(real_images[i]), cmap=plt.cm.gray)
		plt.axis('off')
		plt.savefig(prefix + '_real_' + str(i) + '.png', bbox_inches='tight')
		plt.matshow(np.squeeze(blurred_images[i]), cmap=plt.cm.gray)
		plt.axis('off')
		plt.savefig(prefix + '_blurred_' + str(i) + '.png', bbox_inches='tight')
		plt.matshow(np.squeeze(generated_images[i]), cmap=plt.cm.gray)
		plt.axis('off')
		plt.savefig(prefix + '_generated_' + str(i) + '.png', bbox_inches='tight')


def read_CelebA(sample_size=10000):
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

def down_sample(image, factor):
	shape = image.shape
	down_sampled_image = np.repeat(np.repeat(image[::factor, ::factor, :], factor, axis=0), factor, axis=1)
	return down_sampled_image[:shape[0], :shape[1], :]

def main():
	# mnist()
	CelebA()


main()
