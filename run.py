import numpy as np
import utils
import os
import glob
import multiprocessing
import copy
import diagnostics

import logging
logger = logging.getLogger(__name__)

class SClas():
	
	def __init__(self, name, workdir=None):
		
		self.name = name
		self.is_encoder = False
		self.is_classifier = False
		self.is_classifier_committee = False
		self.is_diagnostic = False

		# Prepare a workdir by default if needed
		if workdir is None:
			self.workdir = "run_%s" % self.name
		else:
			self.workdir = workdir
		
		# Are we restarting some work from an existing workdir ?
		if os.path.isdir(self.workdir):
			logger.info("Detected workdir %s" % self.workdir)
			self.load(self.workdir)
		else:
			self._create_dirs()

	def __str__(self):
		return '%s' % (self.name)
		
	def train_classifier(self, method, traindata, labelsdata=None, strategy="one-vs-all", n=1, 
						ncpu=1,	workdir=None, skipdone=False):
		"""
		Training method for the encoding and the classifier : CAN THIS STAY W/IN ONE METHOD??
		This is a generic method that call the methods in algos/.
		
		:param element: `encoding` or `classifier`, decide which element to train.
		:param method: the specific algorithm to train, must be in algos/element/.
		:param traindata: the training set, for now **ready to use** *FORMAT* ???
		:param labelsdata: the training set labels, if necessary. Only for the classifier, will be
			appended to traindata once it is encoded.
		:param workdir: changes the workdir of the method.
		:param n: number of classifier to train (for committees)
		:param ncpu: number of cpu to use.
		:param skipdone: if the method has already be trained, should retrain ?
		
		:returns: Nothing, saves the final trained algo and the params
		"""

		if not self.is_encoder:
			raise RuntimeError("Encoder is not yet trained, cannot train classifier")
		
		if self.is_classifier and skipdone: 
			logger.info("Skipping classifier training %s, already done." % method)
			return
		
		if labelsdata is None:
			# Nothing to do here, we pass
			pass
		if not np.shape(traindata)[0] == np.shape(labelsdata)[0]:
			raise IndexError("The number of training data does not match the number of labels")
		
		# Append the labels to the data
		traindata = self.encoder.encode(traindata)

		labelsdata = labelsdata.reshape(len(labelsdata), 1)

		traindata = np.concatenate((traindata, labelsdata), axis=1)

		logger.debug("Training method %s" % method)
		method.set_workdir(self._get_dir('algos/classifier'))

		if not workdir is None:
			method.set_workdir(os.path.join(method.workdir, workdir))

		self.classifier = [method] 
		self.is_classifier = True

		comm = [obj for obj in glob.glob(os.path.join(method.workdir, "*")) if os.path.isdir(obj)]
		one = glob.glob(os.path.join(method.workdir, "*.pkl"))
		
		if strategy == "one-vs-all":
			nclas = [None]
			all_traindata = [traindata] 
		elif strategy == "one-vs-one":
			nclas = np.unique(labelsdata)
			all_traindata = []
			for kk in nclas:
				# Changing the array to {0, 1}
				current_label = copy.copy(labelsdata)
				idpos = current_label == kk
				idneg = np.logical_not(current_label == kk)
				current_label[idpos] = 1
				current_label[idneg] = -1

				# Making sure the True value are at the same place as before !
				assert np.all((current_label == 1) == (labelsdata == kk))
				
				traindata[:,-1] = current_label.T
				all_traindata.append(copy.copy(traindata))
				del current_label
				
		else:
			raise RuntimeError("Unknown strategy")

		if n == 1:
			if len(comm) > 0 : 
				raise RuntimeError("Found a committee in the workdir, cowardly refusing to continue")
			
			if strategy == "one-vs-all":
				method.train(traindata)
				method.for_class = None
				method.save()

			elif strategy == "one-vs-one":
				trainparams = [[copy.deepcopy(method), trapa, 0, nclas[ii]] for ii, trapa in zip(range(len(nclas)), all_traindata)]	

				if ncpu == 1: # The single-processing version, much easier to debug !
					self.classifier = map(_trainer_classifier, trainparams)
		
				else: # The simple multiprocessing map is:
					pool = multiprocessing.Pool(processes=ncpu)
					self.classifier = pool.map(_trainer_classifier, trainparams)
					pool.close()
					pool.join()

			
		else:
			if len(one) > 0:
				raise RuntimeError( \
					"Found a single classifier in the workdir, cowardly refusing to continue")
			if len(comm) > n:
				msg = "Found a larger committee in the workdir than what will be computed cowardly refusing to continue"
				raise RuntimeError(msg)
			
			self.is_classifier_committee = True
			trainparams = [[copy.deepcopy(method), trapa, ii, ncl] for ii in range(n) for ncl, trapa in zip(nclas, all_traindata)]
			#print trainparams[0]; exit()
			#for tp in trainparams:
			#	print tp[-1], tp[-2], np.size(np.nonzero(tp[-3][:,-1])) 
			#exit()
			if ncpu == 1: # The single-processing version, much easier to debug !
				self.classifier = map(_trainer_classifier, trainparams)
	
			else: # The simple multiprocessing map is:
				pool = multiprocessing.Pool(processes=ncpu)
				self.classifier = pool.map(_trainer_classifier, trainparams)
				pool.close()
				pool.join()

			
			
		
		
	def train_encoder(self, method, traindata, workdir=None, skipdone=False):
		"""
		Training method for the encoder
		This is a generic method that call the methods in algos/.
		
		:param method: the specific algorithm to train, must be in algos/encoder/.
		:param traindata: the training set, for now **ready to use** *FORMAT* ???
		:param workdir: changes the workdir of the method
		:param skipdone: if the method has already be trained, should retrain ?
		
		:returns: Nothing, saves the final trained algo and the params
		"""

		if self.is_encoder and skipdone: 
			logger.info("Skipping encoder training %s already done." % method)
			return
		
		logger.debug("Training method %s" % method)
		method.set_workdir(self._get_dir('algos/encoder'))

		if not workdir is None:
			method.set_workdir(os.path.join(method.workdir, workdir))
		
		method.train(traindata)
		
		self.encoder = method
		self.is_encoder = True

		method.save()

	def test(self, diagnostic, testdata):
		"""
		Test method for the whole system, according to some diagnostic
		
		:param diagnostic: the specific algorithm to use, must be in algos/diagnostic/.
		:param testdata: the test set ***FORMAT*** ???
		
		:returns: saves the results
		"""
		
	def predict(self, data, treat_catalog=None, workdir=None, **kwargs):
		"""
		Use a method for encoding and a classifier it trained for classifying data.
		
		:param data: ...
		:param treat_catalog: what to do if there is a committee of classifiers. Default: returns
			all catalogues, `mean`: takes the mean of the catalogs, `median` takes the median.
		:param workdir: another workdir than the one in memory
		:returns: a catalog with each classification of the object (in the same order as input)
		"""
		
		if not self.is_classifier and not self.is_encoder:
			raise RuntimeError("Classifier or/and encoder are not trained, can't proceed")
		
		encoded_data = self.encoder.encode(data)

		catalog = []
		uncertainty = []

		for classifier in self.classifier:
			if workdir is not None:
				idclass = classifier.workdir.split("/algos/")[1]
				classifier.change_workdir(os.path.join(workdir, "algos", idclass))
			#print classifier.workdir; print len(self.classifier); exit()
			
			cat, uncer = classifier.predict(encoded_data)
			cat.resize([len(cat)])
			catalog.append(cat)
			uncertainty.append(uncer)
		catalog = np.asarray(catalog)
		uncertainty = np.std(catalog, axis=0)
		
		if treat_catalog is None:
			pass
			# Nothing to do here
		elif treat_catalog == "mean":
			catalog = np.mean(catalog, axis=0)
		elif treat_catalog == "median":
			catalog = np.median(catalog, axis=0)
		elif treat_catalog == "one-vs-one":
			if 'threshold' in kwargs:
				threshold = kwargs['threshold']
			else:
				threshold = 0.9

			for catid, classifier in enumerate(self.classifier):
				cat = catalog[catid]
				idpos = cat >= threshold
				idneg = cat < threshold
				cat[idneg] = -1
				cat[idpos] = classifier.for_class
				catalog[catid] = cat
		else:
			catalog, uncertainty = self.classifier.predict(encoded_data)
			
		if np.shape(catalog)[0] == 1:
			catalog = catalog[0].reshape([len(catalog[0]), 1])
			uncertainty = uncer
		
		return catalog, uncertainty
		
	def save(self, filepath=None):
		"""
		Saves the current instance of sclas in the current stage.
		"""

	def load(self, filepath, algos={"encoder":None, "classifier":None, "diagnostic":None}):
		"""
		Loads and prepare an instance of sclas.
		"""
			
		elements = ["encoder", "classifier", "diagnostic"]
		for element in elements:
			if algos[element] is None:
				filename = "*.pkl"
			else: filename = algos[element]
				
			pathexplored = os.path.join(filepath, "algos", element, filename)
			method = glob.glob(pathexplored)
			comm = [obj for obj in glob.glob(os.path.join(filepath, "algos/%s/*" % element)) 
					if os.path.isdir(obj)]
			
			if element == 'classifier' and (len(method) > 0 or len(comm) > 0):
				
				if len(comm) > 1:
					self.classifier = []
					for classifier in comm :
						try:
							found_file = glob.glob(os.path.join(classifier, "*.pkl"))
							if not len(found_file) == 1:
								msg = "Found multiple files:", found_file
								logger.warn(msg)
								raise IOError()
							classi = utils.readpickle(found_file[0])
							self.classifier.append(classi)
							self.is_classifier = True

						except IOError:
							logger.warn("Found no valid file in %s" % os.path.join(classifier, "*.pkl"))
							
					logger.info("Found %d %ss" % (len(self.classifier), element))
				else:
					self.classifier = [utils.readpickle(method[0])]
					self.is_classifier = True
					
					logger.info("Found a classifier %s" % (self.classifier))
					
				

			elif not element == 'classifier' and len(method) >= 1:
				logger.info("Found %d %s" % (len(method), element))
				
				method = utils.readpickle(method[0])
				
				setattr(self, element, method)
				setattr(self, "is_%s" % element, True)	
			else:
				logger.warning("Found no existing %s matching %s" % (element, pathexplored))
				setattr(self, "is_%s" % element, False)
		
	def info(self):
		"""
		Prints out all the variables
		"""
		import inspect
	
		message = "All variables available for scals %s" % self.name		
		print message
		print '-'*len(message)
		attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
		for a in attributes:
			if (a[0].startswith('__') and a[0].endswith('__')): continue
			print a[0], "=", a[1]
			
	def _get_dir(self, dirname):
		"""
		Returns a string `self.workdir/dirname`.
		
		:param dirname: a string containing the directory name to be appended to workdir
		"""
		return os.path.join(self.workdir, dirname)
	
	def _create_dirs(self):
		"""
		Generate the tree structure needed for saving the data and the sclas config files
		"""
		
		logger.info("Creating output directories")
		utils.mkdir(self.workdir)
		utils.mkdir(self._get_dir('algos'))
		utils.mkdir(self._get_dir('algos/encoder'))
		utils.mkdir(self._get_dir('algos/classifier'))
		utils.mkdir(self._get_dir('data'))
		utils.mkdir(self._get_dir('diagnostics'))
		utils.mkdir(self._get_dir('figures'))
		
		
#===================================================================================================
# Definition of the worker function that multiprocesses the training of the classifier
#===================================================================================================
def _trainer_classifier(params):
	classifier_method, traindata, idscl, class_label = params

	if class_label is None:
		cll = ""
	else:
		cll = "class_%s-" % class_label
	classifier_method.workdir = os.path.join(classifier_method.workdir, "%s%s" % (cll, str(idscl)))
	classifier_method.train(traindata)
	classifier_method.for_class = class_label
	classifier_method.save()
	return classifier_method

