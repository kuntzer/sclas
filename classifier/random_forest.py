"""
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
"""

from sklearn import ensemble
import numpy as np

from .. import method

class ExtraRandomForest(method.Method):
	
	def __init__(self, params):
		"""
		if not 'C' in params:
			params.update({'C': 100.})
		"""
		self.params = dict(params)

		del params['features']
		del params['labels']
		# Models we will use
		self.classifier = ensemble.ExtraTreesRegressor(**params)		
		
	def __str__(self):
		return "ExtraRandomForest from scikit-learn.org"
		
	def train(self, catalog):
		# Training Logistic regression		
		featuresdata = catalog[:,self.params['features']]	
		idlabel = np.array(self.params['features'])[-1] + self.params['labels'] + 1
		labelsdata = catalog[:,idlabel]
		
		labelsdata = labelsdata.reshape(len(labelsdata))

		self.classifier.fit(featuresdata, labelsdata)
		
	def predict(self, data):
		return self.classifier.predict(data), 0.
	
class ExtraRandomForestClassifier(method.Method):
	
	def __init__(self, params):
		"""
		if not 'C' in params:
			params.update({'C': 100.})
		"""
		self.params = dict(params)

		del params['features']
		del params['labels']
		# Models we will use
		self.classifier = ensemble.ExtraTreesClassifier(**params)		
		
	def __str__(self):
		return "ExtraRandomForest from scikit-learn.org"
		
	def train(self, catalog):
		# Training Logistic regression		
		featuresdata = catalog[:,self.params['features']]	
		idlabel = np.array(self.params['features'])[-1] + self.params['labels'] + 1
		labelsdata = catalog[:,idlabel]
		
		labelsdata = labelsdata.reshape(len(labelsdata))

		self.classifier.fit(featuresdata, labelsdata)
		
	def predict(self, data):
		return self.classifier.predict(data), 0.
