from DCGAN import *
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist/", one_hot=True)

def main():
	gan = DCGAN()
	gan.train(mnist.train.images.reshape(-1, 28, 28, 1), mnist.train.images.reshape(-1, 28, 28, 1))

main()
