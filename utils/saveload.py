import numpy as np
import lasagne

def savemodel(network, filename):
	np.savez(filename, *lasagne.layers.get_all_param_values(network))
