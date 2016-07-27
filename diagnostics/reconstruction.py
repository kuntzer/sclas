import pylab as plt
import numpy as np

def reconstruction(encoder, data, labels, n=range(10), show=True):
	components = encoder.encode(data)
	reconstruction = encoder.decode(components)
	
	size = np.int(np.sqrt(np.shape(reconstruction)[1]))
	
	rmse = np.sqrt(reconstruction * reconstruction - data * data) / np.shape(data)[0]
	
	if not show: 
		return rmse
	for i in n:
		plt.figure()
		
		D = data[i].reshape([size, size])
		R = reconstruction[i].reshape([size, size])
		
		ax = plt.subplot(131)
		ax.imshow(D, interpolation="None")
		plt.title("Data, label: %g" % labels[i])
		
		ax = plt.subplot(132)
		ax.imshow(R, interpolation="None")
		plt.title("Reconstructed data")
		
		ax = plt.subplot(133)
		ax.imshow(D-R, interpolation="None")
		plt.title("Residues")
		plt.show()
		
	return rmse