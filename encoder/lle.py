"""
Does the LLE decomposition according to:
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
"""
from sklearn.manifold import LocallyLinearEmbedding
from .. import method

class LLE(method.Method):
	
	def __init__(self, params):
		self.params = params
		self.dec = LocallyLinearEmbedding(**params)
	
	def __str__(self):
		return "LocallyLinearEmbedding"
		
	def train(self, data):
		"""
		Train the NMF on the withened data
		
		:param data: whitened data, ready to use
		"""
		self.dec.fit(data)
	
	def encode(self, data):
		"""
		Encodes the ready to use data
		
		:returns: encoded data with dimension n_components
		"""
		return self.dec.transform(data)
	
	def decode(self, components):
		"""
		Decode the data to return whitened reconstructed data
		
		:returns: reconstructed data
		"""
		return self.dec.inverse_transform(components)
