import numpy as np
import lasagne
from keras.datasets import mnist






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