import matplotlib.pyplot as plt
import numpy as np


def create_image(generate, N=1, M=1, configs=None):
	rimg = generate(np.random.rand(N*M, configs['GIN'], 1, 1).astype('float32')) \
		.astype('float64').reshape(N, M, configs['img_rows'], configs['img_cols']).transpose(0, 2, 1, 3)
	plt.imsave('test.png', rimg.reshape(N*configs['img_rows'], M*configs['img_cols']), cmap='gray')

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
