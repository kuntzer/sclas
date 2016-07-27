'''
"""
Does the PCA decomposition according to:
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
"""
import sklearn.decomposition as dec 
from .. import method

class PCA(method.Method):
	
	def __init__(self, params):
		
		self.params = params
		self.dec = dec.PCA(**params)
	
	def __str__(self):
		return "PCA decomposition"
		
	def train(self, data):
		"""
		Train the PCA on the data
		
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

'''

"""
Does the PCA decomposition according to:
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
"""
import sklearn.decomposition as dec 
from .. import method
from .. import utils as u

import logging
logger = logging.getLogger(__name__)

class PCA(method.Method):
	
	def __init__(self, params):
		
		if 'keep_components' in params:
			self.keep_components = params['keep_components']
			params = u.removekey(params, 'keep_components')
		else:
			self.keep_components = range(params['n_components'])
		self.params = params
		self.dec = dec.PCA(**params)
	
	def __str__(self):
		return "PCA decomposition"
		
	def train(self, data):
		"""
		Train the PCA on the data
		
		:param data: whitened data, ready to use
		"""
		self.dec.fit(data)
	
	def encode(self, data, force_all_components=False):
		"""
		Encodes the ready to use data
		
		:returns: encoded data with dimension n_components
		"""
		if force_all_components:
			return self.dec.transform(data)
		else:
			try:
				return self.dec.transform(data)[:,self.keep_components]
			except:
				logger.warn('%s does not possess "self.keep_components", returning all PCA components' % self.__str__())
				return self.dec.transform(data)
	
	def decode(self, components):
		"""
		Decode the data to return whitened reconstructed data
		
		:returns: reconstructed data
		"""
		return self.dec.inverse_transform(components)
