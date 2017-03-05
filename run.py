import lasagne
import time
import numpy as np


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


def run(X_train, y_train, X_test, y_test, X_val, y_val, num_epochs, train_fn, val_fn, val_fn_gen, G_params,configs=None):

	shuffleset = configs['shuffleset']
	batch_size = configs['batch_size']
	GIN = configs['GIN']

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
