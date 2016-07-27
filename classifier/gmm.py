"""
http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html#sklearn.mixture.GMM
"""

from sklearn.mixture import GMM
import numpy as np

from .. import method

class GaussianMixtureModel(method.Method):
	
	def __init__(self, params):

		self.params = dict(params)

		del params['features']
		del params['labels']
		
		self._set_default(params, 'covariance_type', 'full')
		#self._set_default(params, 'n_iter', 200)

		self.classifier = GMM(**params)
		
	def __str__(self):
		return "Gaussian Mixture Model from scikit-learn.org"
		
	def train(self, catalog):
		featuresdata = catalog[:,self.params['features']]	
		idlabel = np.array(self.params['features'])[-1] + self.params['labels'] + 1
		labelsdata = catalog[:,idlabel]
		
		labelsdata = labelsdata.reshape(len(labelsdata))
		self.all_labels = np.unique(labelsdata)

		self.classifier.fit(featuresdata, labelsdata)
		
	def predict(self, data):
		outcat = self.classifier.predict(data)
		outcat = np.unique(self.all_labels)[outcat]
		return outcat, 0.
