'''
Trains a simple convet on a mixture of real MNIST data
and fake generated data

'''

from keras.datasets import mnist
import theano
import theano.tensor as T
import lasagne
import numpy as np
import time
import matplotlib.pyplot as plt
import pdb

from lasagne.layers import batch_norm

img_rows, img_cols = 28,28 # Input image dimensions
batch_size = 128
nb_classes = 1
nb_epoch = 12
nb_filters = 32 # Number of convolution filters
pool_size = (2,2)
kernel_size = (3,3)
GIN = 32
shuffleset = False

def load_dataset():
    # Load the data

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    X_train = X_train.reshape(-1, 1, img_rows, img_cols)
    #    y_train = X_train.reshape(-1, 1, img_rows, img_cols)
    X_test = X_test.reshape(-1, 1, img_rows, img_cols)
    #    y_test = X_train.reshape(-1, 1, img_rows, img_cols)

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    y_train = np.ones(y_train.shape).astype('int32')
    y_test = np.ones(y_test.shape).astype('int32')
    y_val = np.ones(y_val.shape).astype('int32')

    return (X_train, y_train, X_test,y_test, X_val, y_val)

def classifier(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, img_rows, img_cols),
                                        input_var=input_var)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=nb_filters, filter_size=kernel_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=pool_size)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=nb_filters, filter_size=kernel_size,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=pool_size)
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.25),
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=nb_classes,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    network = lasagne.layers.ReshapeLayer(network, (-1, nb_classes))

    return network


def synthesiser(input_var=None):

    NB = GIN
    network = lasagne.layers.InputLayer(shape=(None, NB, 1, 1), input_var=input_var)
    print('L0:'+str(lasagne.layers.get_output_shape(network)))
    network = batch_norm(lasagne.layers.DenseLayer(
            incoming=network,
            num_units=1024,
            nonlinearity=lasagne.nonlinearities.rectify
    ))
    print('L1:'+str(lasagne.layers.get_output_shape(network)))
    network = lasagne.layers.ReshapeLayer(network, (-1, 1024))
    network = batch_norm(lasagne.layers.DenseLayer(
                incoming=network,
                num_units=128*7*7,
                nonlinearity=lasagne.nonlinearities.rectify
        ))
    print('L2:'+str(lasagne.layers.get_output_shape(network)))
    network = lasagne.layers.ReshapeLayer(network, (-1, 128, 7, 7))
    network = batch_norm(lasagne.layers.Deconv2DLayer(
        incoming=network,
        num_filters= 64,
        filter_size=(3,3),
        stride=2,
        nonlinearity=lasagne.nonlinearities.rectify
   ))
    print('L3:'+str(lasagne.layers.get_output_shape(network)))
    network = lasagne.layers.Deconv2DLayer(
            incoming=lasagne.layers.dropout(network, p=.25),
            num_filters=1,
            filter_size=(2,2),
            stride=2,
            crop='full',
            nonlinearity=lasagne.nonlinearities.rectify
    )

    print('L4:'+str(lasagne.layers.get_output_shape(network)))
    network = lasagne.layers.ReshapeLayer(network, (-1, 1, img_rows, img_cols))
    print('L5:'+str(lasagne.layers.get_output_shape(network)))

    return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

    yield (inputs[excerpt], targets[excerpt])

def run(X_train, y_train, X_test, y_test, X_val, y_val, num_epochs, train_fn, val_fn, val_fn_gen, G_params):
    print("Starting training...")
    # We iterate over epochs:
    train_err = {}
    valid_err = {}
    valid_acc = {}
    train_err['gen'] = []
    train_err['discrim'] = []
    valid_err['gen'] = []
    valid_err['discrim'] = []
    valid_acc['real'] = []
    valid_acc['synth'] = []

    G_weights = {}

    G_weights['W1'] = []
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        epoch_loss = {}
        epoch_loss['gen'] = 0
        epoch_loss['discrim'] = 0
        train_batches = 0
        start_time = time.time()


        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=shuffleset):
            inputs, targets= batch
            Gseed = lasagne.utils.floatX(np.random.rand(len(inputs), GIN, 1,1))
            loss = np.array(train_fn(Gseed, inputs))
            epoch_loss['gen'] += loss[0]
            epoch_loss['discrim'] += loss[1]

            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        gen_err = 0
        gen_acc = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=shuffleset):
            inputs, targets = batch
#            err, acc = val_fn(inputs)
            Gseed = lasagne.utils.floatX(np.random.rand(len(inputs), GIN, 1,1))
            err, acc = val_fn(inputs)
            val_err += err
            val_acc += acc
            err, acc = val_fn_gen(Gseed)
            gen_err += err
            gen_acc += acc
            val_batches += 1

        train_err['gen'].append(epoch_loss['gen'])
        train_err['discrim'].append(epoch_loss['discrim'])
        valid_err['discrim'].append(val_err / val_batches)
        valid_err['gen'].append(gen_err / val_batches)
        valid_acc['real'].append(val_acc / val_batches * 100)
        valid_acc['synth'].append(gen_acc / val_batches * 100)
        G_weights['W1'].append(np.median(G_params[2].get_value()))
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
        print("  Discriminator[Train] loss:\t\t\t{:.6f}".format(epoch_loss['discrim'] / train_batches))
        print("  Discriminator[Valid] loss:\t\t\t{:.6f}".format(val_err / val_batches))
        print("  Generator[Train] loss:\t\t\t{:.6f}".format(epoch_loss['gen'] / train_batches))
        print("  Discriminator[real] acc:\t\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
        print("  Discriminator[synthetic] acc\t\t\t{:.2f} %".format(
                gen_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=shuffleset):
        inputs, targets = batch
        err, acc = val_fn(inputs)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))

    return train_err, valid_err, valid_acc, G_weights

def main(num_epochs=500):
    # https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56
    # Prepare Theano variables for inputs. We don't need targets as we'll set them manually
    C_in = T.tensor4('inputs')
    G_in = T.tensor4('random')

    # Load the data
    (X_train, y_train, X_test,y_test, X_val, y_val) = load_dataset()

    # Classifier
    C_network = classifier(C_in)
    C_out = lasagne.layers.get_output(C_network)
    C_params = lasagne.layers.get_all_params(C_network, trainable=True)

    # Define the synthesiser function (that tries to create 'real' images)
    G_network = synthesiser(G_in)
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
        train_fn, val_fn, val_gen_fn, G_params)

    return generate, lossplots

def create_image(generate, N=1, M=1):
    rimg = generate(np.random.rand(N*M, GIN, 1, 1).astype('float32'))\
        .astype('float64').reshape(N, M, img_rows, img_cols).transpose(0, 2, 1, 3)
    plt.imsave('test.png', rimg.reshape(N*img_rows, M*img_cols), cmap='gray')

    #Image.fromarray((255*rimg/np.max(rimg[:])).astype('uint8')).save('test.jpeg')
    return None


def plotloss(lossplots):
    train_err, val_err, val_acc, GW = lossplots
    plt.plot(train_err['discrim'])
    plt.plot(train_err['gen'])
    plt.plot(val_err['discrim'])
    plt.plot(val_err['gen'])

    plt.legend(['Discriminator[Training][Real]', 'Discriminator[Training][Synth]',
                'Discriminator[Validation][Real]', 'Discriminiator[Validation][Synth]'])
    plt.title(' Loss plots')
    plt.show()
    plt.plot(val_acc['real'])
    plt.plot(val_acc['synth'])
    plt.legend(['% Accuracy on real data', '% Accuracy on synthetic data'])
    plt.show()

    plt.plot(GW['W1'])
    plt.show()





if __name__=='__main__':
    generate,lossplots = main(num_epochs=500)
    create_image(generate, 6, 7)
    plotloss(lossplots)
    pdb.set_trace()



