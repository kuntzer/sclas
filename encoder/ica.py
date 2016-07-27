"""
Does the ICA decomposition according to:
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
"""
from sklearn.decomposition import FastICA
from .. import method

class ICA(method.Method):
	
	def __init__(self, params):
		self.params = params
		self.ica = FastICA(**params)
	
	def __str__(self):
		return "FastICA"
		
	def train(self, data):
		"""
		Train the FastICA on the withened data
		
		:param data: whitened data, ready to use
		"""
		self.ica.fit(data)
	
	def encode(self, data):
		"""
		Encodes the ready to use data
		
		:returns: encoded data with dimension n_components
		"""
		return self.ica.transform(data)
	
	def decode(self, components):
		"""
		Decode the data to return whitened reconstructed data
		
		:returns: reconstructed data
		"""
		return self.ica.inverse_transform(components)
