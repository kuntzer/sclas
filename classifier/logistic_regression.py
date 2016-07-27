"""
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""

from sklearn import linear_model
import numpy as np

from .. import method

class LogisiticRegression(method.Method):
	
	def __init__(self, params):
		if not 'C' in params:
			params.update({'C': 100.})

		self.params = dict(params)

		del params['features']
		del params['labels']
		# Models we will use
		self.classifier = linear_model.LogisticRegression(**params)		
		
	def __str__(self):
		return "Logisitic Regression from scikit-learn.org"
		
	def train(self, catalog):
		# Training Logistic regression		
		featuresdata = catalog[:,self.params['features']]	
		idlabel = np.array(self.params['features'])[-1] + self.params['labels'] + 1
		labelsdata = catalog[:,idlabel]
		
		labelsdata = labelsdata.reshape(len(labelsdata))

		self.classifier.fit(featuresdata, labelsdata)
		
	def predict(self, data):
		return self.classifier.predict(data), 0.
