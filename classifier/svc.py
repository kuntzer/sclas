"""
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""

from sklearn import svm
import numpy as np

from .. import method

class SupportVectorClassifier(method.Method):
	
	def __init__(self, params):
		
		self.params = dict(params)
		
		#if not 'gamma' in params:
		#	params.update({'gamma': 0.001})
		
		del params['features']
		del params['labels']
	
		# Models we will use
		self.classifier = svm.SVC(**params)		
		
	def __str__(self):
		return "Support Vector Classifier from scikit-learn.org"
		
	def train(self, catalog):
		# Training SVC
		featuresdata = catalog[:,self.params['features']]	
		idlabel = np.array(self.params['features'])[-1] + self.params['labels'] + 1
		labelsdata = catalog[:,idlabel]
		
		labelsdata = labelsdata.reshape(len(labelsdata))

		self.classifier.fit(featuresdata, labelsdata)
		
	def predict(self, data):
		return self.classifier.predict(data), 0.
