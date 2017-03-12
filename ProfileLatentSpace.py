"""
In order to profile the latent space of the distribution we do the following:
Train an MNIST digit classifier
Generate some digits using the generative model

"""
from networks import synthesiser, classifier
import theano
import lasagne
import theano.tensor as T
import numpy as np
from utils.saveload import initnetwork
from utils.plotting import create_image


configs = {}
configs['img_rows'], configs['img_cols'] = 28,28 # Input image dimensions
configs['batch_size'] = 128
configs['GIN'] = 100
configs['shuffleset'] = False

def main(configs=configs):

	# Load and initialise the networks
	G_in = T.tensor4('random')
	C_in = T.tensor4('image')
	G_network = synthesiser(G_in, configs=configs)
	C_network = classifier(C_in)
	initnetwork(G_network, 'generator.npz')
	generate = theano.function([G_in], lasagne.layers.get_output(G_network,deterministic=True))
	classify = theano.function([C_in], T.argmax(lasagne.layers.get_output(C_network, deterministic=True), axis=1))
	create_image(generate, 6, 7, name='test_seed2.png', seed=2, configs=configs)

	np.random.seed(seed=42)
	rimg = generate(np.random.rand(6*7, configs['GIN'], 1, 1).astype('float32'))
	X = classify(rimg).reshape(6, 7)
	print(X)


if __name__=='__main__':
	main()