import numpy as np
from .. import method

import logging
logger = logging.getLogger(__name__)

class Confusion_Matrix(method.Method):
	"""
	Class to compute the classical tests for binary classification.
	See http://en.wikipedia.org/wiki/Confusion_matrix
	"""
	
	def __init__(self, catalog, truth, unknown_cat_name=None):
		"""
		:param catalog: a 1D array containing only the classified output.
		:param truth: the corresponding truth.
		:param unknown_cat_name: the name of the possible category `unknown` i.e. not seeing in the 
			training set or critical failure
		:param unknown_cat_id: the id of the possible category `unknown` i.e. not seeing in the 
			training set or critical failure
		"""
		
		self.classes = np.unique(truth)
		self.n_classes = np.size(self.classes)	
		
		#if unknown_cat_id is None != unknown_cat_name is None:
		#	raise ValueError("There is an unknown category, but the name or the id is missing!")
		
		if not unknown_cat_name is None and not unknown_cat_name in self.classes:
			self.classes = np.append(self.classes, unknown_cat_name)
			unknown_cat_id = len(self.classes)-1
			self.n_classes += 1
		elif unknown_cat_name in self.classes:
			unknown_cat_id = np.where(self.classes == unknown_cat_name)[0][0]
		
		matrix = np.zeros([self.n_classes, self.n_classes], dtype=int)
		
		self.true_positive = np.zeros(self.n_classes)
		self.true_negative = np.zeros_like(self.true_positive)
		self.false_positive = np.zeros_like(self.true_positive)
		self.false_negative = np.zeros_like(self.true_positive)
		self.n_test = np.size(catalog)
	
		for ii, c in enumerate(self.classes):
			# Get the positive condition (ie, truth is 1)
			condition_positive = np.where(truth == c)
			try:
				cp = catalog[condition_positive] - c
			except:
				print condition_positive, catalog
				logger.critical("There's an error here")
				raise IndexError()
			# Classification outcome is 0
			self.false_negative[ii] = np.count_nonzero(cp)
			# Classification outcome is 1
			self.true_positive[ii] = np.size(cp) - self.false_negative[ii]
			
			# Get the negative condition (ie, truth is 0)
			condition_negative = np.where(truth != c)
			cn = catalog[condition_negative] - c
			# Classification outcome is 0
			self.true_negative[ii] = np.count_nonzero(cn)
			# Classification outcome is 1
			self.false_positive[ii] = np.size(cn) - self.true_negative[ii]

		# This is slower but can account for more classifications ids than histogram2d
		for eff_class, tru_class in zip(catalog, truth):
			ii = np.where(tru_class == self.classes)[0][0]
			try:
				jj = np.where(eff_class == self.classes)[0][0]
			except IndexError:
				if not unknown_cat_name is None:
					jj = unknown_cat_id
				else:
					logger.critical(("Error: unknown output classification : %g ; Available classes:" 
								% (eff_class), self.classes))
					raise IndexError()
			matrix[ii, jj] += 1

		self.matrix = matrix
			
	def accuracy(self):
		"""
		Accuracy evaluation diagnostics, good for somewhat equally distributed class populations
		http://en.wikipedia.org/wiki/Accuracy_and_precision
		Accuracy = (true positive + true negative) / total
		"""
		return (self.true_negative + self.true_positive) / (self.n_test)
	
	def precision(self):
		"""
		Precision evaluation diagnostics, good for somewhat equally distributed class populations
		http://en.wikipedia.org/wiki/Accuracy_and_precision
		Precision = true positive / (true positive + false positive)
		"""
		det = self.false_positive + self.true_positive
		det[det == 0] = 1
		return self.true_positive / det
	
	def contamination(self):
		"""
		contamination = false positive / (true positive + false positive)
		"""
		det = self.false_positive + self.true_positive
		det[det == 0] = 1
		return self.false_positive / det
	
	def recall(self):
		"""
		True positive rate (TPR, Sensitivity, Recall)
		Recall = true positive / (false negative + true positive)
		"""
		det = self.false_negative + self.true_positive
		det[det == 0] = 1
		return self.true_positive / det
	
	def completeness(self):
		"""
		Same as recall
		"""
		return self.recall()
	
	def f1score(self):
		"""
		See http://en.wikipedia.org/wiki/F1_score
		"""
		prec = self.precision()
		reca = self.recall()
		det = prec + reca 
		det[det == 0] = 1
		# Compute the F1 score
		f1 = 2 * prec * reca / det
		
		# If there is a F1 score == 0 it might be that there is no representative of this class.
		if np.any(f1 == 0.):
			for ii, f1ii in enumerate(f1):
				if f1ii > 0: continue
				countf = self.matrix[:,ii].sum() + self.matrix[ii].sum()
				if countf == 0:
					f1[ii] = np.nan
					
			#print f1
		return f1
	
	def gmeasure(self):
		"""
		While the F-measure is the harmonic mean of Recall and Precision, the G-measure 
			is the geometric mean.
		"""
		return np.sqrt(self.precision() * self.recall())
	
	def get_matrix(self):
		return self.matrix
	