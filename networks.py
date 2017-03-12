import lasagne
from lasagne.layers import InputLayer, batch_norm, DenseLayer, Conv2DLayer, Deconv2DLayer
from lasagne.nonlinearities import sigmoid, rectify, LeakyRectify, identity
from lasagne.layers.dnn import batch_norm_dnn as batch_norm
nb_classes = 1
nb_filters = 32 # Number of convolution filters
pool_size = (2,2)
kernel_size = (3,3)

def synthesiser(input_var=None, configs=None):

	network = lasagne.layers.InputLayer(shape=(None, configs['GIN'], 1, 1), input_var=input_var)
	print('L0:'+str(lasagne.layers.get_output_shape(network)))
	network = batch_norm(DenseLayer(
			incoming=network,
			num_units=1024,
	))
	print('L1:'+str(lasagne.layers.get_output_shape(network)))
	network = lasagne.layers.ReshapeLayer(network, (-1, 1024))
	# Project, reshape
	network = batch_norm(DenseLayer(
			incoming=network,
			num_units=128*7*7,
	))
	print('L2:'+str(lasagne.layers.get_output_shape(network)))
	network = lasagne.layers.ReshapeLayer(network, (-1, 128, 7, 7))
	# Two fractional-stride convolutions
	network = batch_norm(Deconv2DLayer(
			incoming=network,
			num_filters=64,
			filter_size=(3,3),
			stride=2,
	))
	print('L3:'+str(lasagne.layers.get_output_shape(network)))
	network = lasagne.layers.Deconv2DLayer(
			incoming=network,
			num_filters=1,
			filter_size=(2,2),
			stride=2,
			crop='full',
			output_size=28,
			nonlinearity=lasagne.nonlinearities.sigmoid
	)

	print('L4:'+str(lasagne.layers.get_output_shape(network)))
	network = lasagne.layers.ReshapeLayer(network, (-1, 1, configs['img_rows'], configs['img_cols']))
	print('L5:'+str(lasagne.layers.get_output_shape(network)))

	return network


def discriminator(input_var=None, configs=None):

	lrelu = LeakyRectify(0.2)

	network = InputLayer(shape=(None, 1, configs['img_rows'], configs['img_cols']), input_var=input_var)
	network = batch_norm(Conv2DLayer(
			network,
			num_filters=64,
			filter_size=(5,5),
			stride=2,
			nonlinearity=lrelu,
			W=lasagne.init.GlorotUniform()
	))
#	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))
	network = batch_norm(lasagne.layers.Conv2DLayer(
			network,
			num_filters=128,
			filter_size=5,
			stride=2,
			nonlinearity=lrelu
	))
#	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))
	network = batch_norm(DenseLayer(incoming=lasagne.layers.dropout(network, p=0.25),num_units=1024,nonlinearity=lrelu))
	network = DenseLayer(
			incoming=network,
	        num_units=1,
			nonlinearity=sigmoid,
	)

	network = lasagne.layers.ReshapeLayer(network, (-1, nb_classes))

	return network

def classifier(input_var=None):
	# As a third model, we'll create a CNN of two convolution + pooling stages
	# and a fully-connected hidden layer in front of the output layer.

	# Input layer, as usual:
	network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
	                                    input_var=input_var)
	# This time we do not apply input dropout, as it tends to work less well
	# for convolutional layers.

	# Convolutional layer with 32 kernels of size 5x5. Strided and padded
	# convolutions are supported as well; see the docstring.
	network = lasagne.layers.Conv2DLayer(
			network, num_filters=32, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform())
	# Expert note: Lasagne provides alternative convolutional layers that
	# override Theano's choice of which implementation to use; for details
	# please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

	# Max-pooling layer of factor 2 in both dimensions:
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
	network = lasagne.layers.Conv2DLayer(
			network, num_filters=32, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	# A fully-connected layer of 256 units with 50% dropout on its inputs:
	network = lasagne.layers.DenseLayer(
			lasagne.layers.dropout(network, p=.5),
			num_units=256,
			nonlinearity=lasagne.nonlinearities.rectify)

	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
	network = lasagne.layers.DenseLayer(
			lasagne.layers.dropout(network, p=.5),
			num_units=10,
			nonlinearity=lasagne.nonlinearities.softmax)

	return network
