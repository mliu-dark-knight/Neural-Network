import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from GMM import *


def main():
	img = mpimg.imread('mountains.png')[:, :, :3]
	img_reshape = np.reshape(img, (-1, 3))
	for m in [3, 5, 10]:
		gmm = GMM(m=m)
		gmm.fit(img_reshape, 10)
		img_cluster = gmm.clustering(img_reshape)
		img_cluster = np.reshape(img_cluster, (img.shape[0], img.shape[1], img.shape[2]))
		plt.imshow(img_cluster)
		plt.title("m = %d" % m)
		plt.show()

main()
