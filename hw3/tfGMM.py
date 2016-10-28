import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.contrib.factorization.python.ops.gmm import GMM
from tensorflow.contrib.factorization.python.ops.kmeans import KMeansClustering as KMeans
from tensorflow.contrib.factorization.python.ops import gmm_ops
from tensorflow.contrib.learn.python.learn.utils import checkpoints

def weights(model):
	return checkpoints.load_variable(model.model_dir, 'Variable')


def main():
	img = mpimg.imread('corgi.png')[:, :, :3]
	img_reshape = np.reshape(img, (-1, 3))
	for m in [3, 5, 10]:
		gmm = GMM(m, steps=10)
		gmm.fit(img_reshape)
		print("w:")
		print(weights(gmm))
		print("mu:")
		print(gmm.clusters())
		print("sigma:")
		print(gmm.covariances())
		clusters = gmm.clusters()
		img_cluster = clusters[gmm.predict(img_reshape)]
		img_cluster = np.reshape(img_cluster, (img.shape[0], img.shape[1], img.shape[2]))
		plt.imshow(img_cluster)
		plt.title("m = %d" % m)
		plt.show()

main()