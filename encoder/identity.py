"""
Returns the identity
"""
from .. import method

class identity(method.Method):
	
	def __init__(self, params):
		self.params = params
	
	def __str__(self):
		return "Identity"
		
	def train(self, data):
		"""
		Just here for consistency
		"""
		pass
	
	def encode(self, data):
		"""
		return the data
		"""
		return data
	
	def decode(self, components):
		"""
		return the components
		"""
		return components
