"""
Applies principles of deep learning for dimensionality reduction.
Using the pylae package (git clone https://kuntzer@git.epfl.ch/repo/pylae.git)  
"""
import pylae
from .. import method


import logging
logger = logging.getLogger(__name__)

class AutoEncoder(method.Method):
	"""
	Nice explanation needed here and a list of all params with the mandatory ones signaled
	"""
	#TODO: Write docstring
	
	def __init__(self, params):
		
		# Making sure a few things are in params
		self._set_default(params, 'rbm_type', 'gd', "No rbm_type was given, assuming gd")
		self._set_default(params, 'verbose', False)
		# Those are needed for pre_train
		self._set_default(params, 'pre_learn_rate', {'SIGMOID':0.1, 'LINEAR':0.01})
		self._set_default(params, 'learn_rate', 0.13)
		self._set_default(params, 'momentum_rate', 0.85)
		self._set_default(params, 'initialmomentum', 0.55)
		self._set_default(params, 'finalmomentum', 0.9)
		self._set_default(params, 'pre_iterations', 2000)
		self._set_default(params, 'iterations', 1000)
		self._set_default(params, 'mini_batch', 100)
		self._set_default(params, 'pre_regularisation', 0.001)
		self._set_default(params, 'regularisation', 0.0)
		self._set_default(params, 'max_epoch_without_improvement', 30)
		self._set_default(params, 'early_stop', 30)
			
		self.params = params
		self.ae = pylae.autoencoder.AutoEncoder(rbm_type=self.params['rbm_type'])

	def __str__(self):
		return "AutoEncoder as implemented in pylae"
		
	def train(self, data):
		"""
		Train the PCA on the withened data
		
		:param data: whitened data, ready to use
		"""
		#===========================================================================================
		# First a layer-wise, greedy pre-training
		#===========================================================================================
		# Those must be in self.params
		architecture = self.params['architecture']
		layers_type = self.params['layers_type']
		# Load the other params
		# TODO : is there a better way to do this ? Certainly !
		learn_rate = self.params['pre_learn_rate']
		initialmomentum = self.params['initialmomentum']
		finalmomentum = self.params['finalmomentum']
		iterations = self.params['pre_iterations']
		mini_batch = self.params['mini_batch']
		regularisation = self.params['pre_regularisation']
		max_epoch_without_improvement = self.params['max_epoch_without_improvement']
		early_stop = self.params['early_stop']
		
		# This will train layer by layer the network by minimising the error
		self.ae.pre_train(data, architecture, layers_type, learn_rate, initialmomentum, 
			finalmomentum, iterations, mini_batch, regularisation, max_epoch_without_improvement, 
			early_stop)
		
		#===========================================================================================
		# Then, we do a back-propagation to finely tune the weights
		#===========================================================================================
		iterations = self.params['iterations']
		learn_rate = self.params['learn_rate']
		momentum_rate = self.params['momentum_rate']
		regularisation = self.params['regularisation']
		
		self.ae.backpropagation(data, iterations, learn_rate, momentum_rate, 
							max_epoch_without_improvement, regularisation, early_stop)
	
	def encode(self, data):
		"""
		Encodes the ready to use data
		
		:returns: encoded data with dimension n_components
		"""
		return self.ae.encode(data)
	
	def decode(self, components):
		"""
		Decode the data to return whitened reconstructed data
		
		:returns: reconstructed data
		"""
		return self.ae.decode(components)
	
