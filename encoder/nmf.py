"""
Does the NMF decomposition according to:
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
"""
from sklearn.decomposition import ProjectedGradientNMF
from .. import method

class NMF(method.Method):
	
	def __init__(self, params):
		self.params = params
		self.dec = ProjectedGradientNMF(**params)
	
	def __str__(self):
		return "Non-Negative matrix factorization by Projected Gradient (NMF)"
		
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
