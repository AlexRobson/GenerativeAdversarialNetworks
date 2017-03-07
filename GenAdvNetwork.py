'''
Trains a simple convet on a mixture of real MNIST data
and fake generated data

'''

import theano
import theano.tensor as T
import lasagne
import numpy as np
import pdb

from networks import synthesiser, discriminator
from run import run
from utils.plotting import create_image, plotloss
from utils.saveload import savemodel, load_dataset

configs = {}
configs['img_rows'], configs['img_cols'] = 28,28 # Input image dimensions
configs['batch_size'] = 128
configs['GIN'] = 100
configs['shuffleset'] = False




def main(num_epochs=500, configs=configs):
    # https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56
    # Prepare Theano variables for inputs. We don't need targets as we'll set them manually
    C_in = T.tensor4('inputs')
    G_in = T.tensor4('random')

    # Load the data
    (X_train, y_train, X_test,y_test, X_val, y_val) = load_dataset(configs=configs)

    # Classifier
    C_network = discriminator(C_in, configs=configs)
    C_out = lasagne.layers.get_output(C_network)
    C_params = lasagne.layers.get_all_params(C_network, trainable=True)

    # Define the synthesiser function (that tries to create 'real' images)
    G_network = synthesiser(G_in, configs=configs)
    G_params = lasagne.layers.get_all_params(G_network, trainable=True)

    real_out = lasagne.layers.get_output(C_network)
    # fake_out, second arg in get_output is optional inputs to pass through to C_network
    fake_out = lasagne.layers.get_output(C_network,
                                         lasagne.layers.get_output(G_network),
                                         deterministic=True)

    # Define the objective, updates, and training functions
    # Cost = Fakes are class=1, so for generator target is for all to be identified as real (0)
    eps = 1e-10
    alfa = 1-1e-5
    G_obj = lasagne.objectives.binary_crossentropy((fake_out+eps)*alfa, 1).mean()
    # Cost = Discriminator needs real = 0, and identify fakes as 1
    C_obj = lasagne.objectives.binary_crossentropy((real_out+eps)*alfa, 1).mean()+\
        lasagne.objectives.binary_crossentropy((fake_out+eps)*alfa, 0).mean()


    updates = lasagne.updates.adam(
            C_obj, C_params, learning_rate=2e-4, beta1=0.5)
    updates.update(lasagne.updates.adam(
        G_obj, G_params, learning_rate=2e-4, beta1=0.5)
    )
    train_fn = theano.function([G_in, C_in],
                               [G_obj,C_obj],
                               updates=updates,
                               name='training')

    # Create the theano functions
    classify = theano.function([C_in], C_out)
    generate = theano.function([G_in], lasagne.layers.get_output(G_network,deterministic=True))

    # The test prediction is running the discriminator deterministically. All the validation set are
    # real, so the cost is when identifiying as fake (1)
    test_prediction = lasagne.layers.get_output(C_network, deterministic=True)
    test_loss = lasagne.objectives.binary_crossentropy(test_prediction, 1)
    test_loss = test_loss.mean()
    test_acc = T.mean(test_prediction>0.5,
                      dtype=theano.config.floatX)

    # Loss is from the perspective of the discriminator, so the target is for all values to be labelled true (0)
    test_generator = lasagne.layers.get_output(C_network,
                                               lasagne.layers.get_output(G_network, deterministic=True),
                                               deterministic=True)
    test_loss_gen = lasagne.objectives.binary_crossentropy(test_generator, 0).mean()
    test_acc_gen = T.mean(test_generator<0.5, dtype=theano.config.floatX)

    # Compile the training and validation functions
    val_fn = theano.function([C_in], [test_loss, test_acc])
    val_gen_fn = theano.function([G_in], [test_loss_gen, test_acc_gen])
#    pdb.set_trace()
    # Run
    lossplots = run(X_train, y_train,
        X_test, y_test,
        X_val, y_val,
        num_epochs,
        train_fn, val_fn, val_gen_fn, G_params, configs=configs)

    networks = {}
    networks['generator'] = G_network
    networks['discriminator'] = C_network

    return generate, networks, lossplots


if __name__=='__main__':
    generate,networks, lossplots = main(num_epochs=500, configs=configs)
    savemodel(networks['generator'], 'generator.npz')
    savemodel(networks['discriminator'], 'discriminator.npz')
    create_image(generate, 6, 7, configs=configs)
    plotloss(lossplots)



