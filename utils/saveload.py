import numpy as np
import lasagne
import os, sys
from keras.datasets import mnist
import pdb


def load_mnist():
	# We first define a download function, supporting both Python 2 and 3.
	if sys.version_info[0] == 2:
		from urllib import urlretrieve
	else:
		from urllib.request import urlretrieve

	def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
		print("Downloading %s" % filename)
		urlretrieve(source + filename, filename)

	# We then define functions for loading MNIST images and labels.
	# For convenience, they also download the requested files if needed.
	import gzip

	def load_mnist_images(filename):
		if not os.path.exists(filename):
			download(filename)
		# Read the inputs in Yann LeCun's binary format.
		with gzip.open(filename, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=16)
		# The inputs are vectors now, we reshape them to monochrome 2D images,
		# following the shape convention: (examples, channels, rows, columns)
		data = data.reshape(-1, 1, 28, 28)
		# The inputs come as bytes, we convert them to float32 in range [0,1].
		# (Actually to range [0, 255/256], for compatibility to the version
		# provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
		return data / np.float32(256)

	def load_mnist_labels(filename):
		if not os.path.exists(filename):
			download(filename)
		# Read the labels in Yann LeCun's binary format.
		with gzip.open(filename, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=8)
		# The labels are vectors of integers now, that's exactly what we want.
		return data

	froot = '/home/alex/Datasets/MNIST/'
	# We can now download and read the training and test set images and labels.
	X_train = load_mnist_images(froot+'train-images-idx3-ubyte.gz')
	y_train = load_mnist_labels(froot+'train-labels-idx1-ubyte.gz')
	X_test = load_mnist_images(froot+'t10k-images-idx3-ubyte.gz')
	y_test = load_mnist_labels(froot+'t10k-labels-idx1-ubyte.gz')

	# We reserve the last 10000 training examples for validation.
	X_train, X_val = X_train[:-10000], X_train[-10000:]
	y_train, y_val = y_train[:-10000], y_train[-10000:]
	# We just return all the arrays in order, as expected in main().
	# (It doesn't matter how we do this as long as we can read them again.)
	return X_train, y_train, X_val, y_val, X_test, y_test



def savemodel(network, filename):
	np.savez(filename, *lasagne.layers.get_all_param_values(network))


def load_dataset(configs=None):
	# Load the data

	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255

	X_train = X_train.reshape(-1, 1, configs['img_rows'], configs['img_cols'])
	#    y_train = X_train.reshape(-1, 1, img_rows, img_cols)
	X_test = X_test.reshape(-1, 1, configs['img_rows'], configs['img_cols'])
	#    y_test = X_train.reshape(-1, 1, img_rows, img_cols)

	# We reserve the last 10000 training examples for validation.
	X_train, X_val = X_train[:-10000], X_train[-10000:]
	y_train, y_val = y_train[:-10000], y_train[-10000:]

	y_train = np.ones(y_train.shape).astype('int32')
	y_test = np.ones(y_test.shape).astype('int32')
	y_val = np.ones(y_val.shape).astype('int32')

	return (X_train, y_train, X_test,y_test, X_val, y_val)

def initnetwork(network, filename):
	with np.load(filename) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]

	lasagne.layers.set_all_param_values(network, param_values)