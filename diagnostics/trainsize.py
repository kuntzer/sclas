import numpy as np
import multiprocessing
import copy
import confusion
from .. import utils

import logging
logger = logging.getLogger(__name__)

def test_trainsize(ml, metric, traindata, traintruth, testdata, testturth, fractions=[0.2, 0.4, 0.6, 0.8, 1.], ncpu=1):
	"""
	Computes a metric for different training sizes and plots it. 
	
	:param ml: the classifier instance
	:param traindata: already encoded data, training set
	:param traintruth: training truth
	:param testdata: same but for testing set
	:param testturth: 
	:param metric: A string of the name of one of testing.Testing method
	:param fractions: the fractions of the traindata to test
	:param ncpu: how many cpus to use ?
	
	All remaining arguments are passed to `pyplot.plot`.
	"""
	# Do this here to avoid X11 import error on certain machines
	import matplotlib.pyplot as plt
	from .. import plots
	
	# TODO: wtf about the hang up issue when ncpu=8 ??? 
	
	logger.info("Beginning the test on different training sample size")
	
	trainparams = [[f, copy.deepcopy(ml), \
				copy.deepcopy(traindata[:np.int(f * np.shape(traindata)[0])]), \
				copy.deepcopy(traintruth[:np.int(f * np.shape(traintruth)[0])]), \
				copy.deepcopy(testdata[:np.int(f * np.shape(testdata)[0])]), \
				copy.deepcopy(testturth[:np.int(f * np.shape(testturth)[0])]), \
				metric
				]\
				 for f in fractions]
	

	if ncpu == 1: # The single-processing version, much easier to debug !
		res = map(_trainer_size, trainparams)

	else: # The simple multiprocessing map is:
		pool = multiprocessing.Pool(processes=ncpu)
		res = pool.map(_trainer_size, trainparams)
		pool.close()
		pool.join()
	
	res = np.sort(np.asarray(res), axis=0)
	errors_training_size = res
	fractions_training_size = np.asarray(fractions)
		
	mykwargs = {"marker":"*", "ms":5, "ls":"--", "alpha":1.}

	print fractions_training_size
	print 'training', errors_training_size[:,1]
	print 'test', errors_training_size[:,2]
	
	fig = plt.figure()
	ax = plt.subplot(111)
	
	xxs = [fractions_training_size * np.shape(traindata)[0]] * 2
	yys = [errors_training_size[:,1], errors_training_size[:,2]]
	labels = ["Training set", "Test set"]
	colors = ["blue", "green"]
	
	plots.plot(ax, xxs, yys, x_name="Training set size", y_name="%s" % metric, labels=labels, colors=colors, **mykwargs)

	return fig, ax

def _trainer_size(params):
	"""
	Worker function for :func:`test_training_size`. It trains a ML and computes the training and 
		validation error
	
	:param params: a list containing [fraction of data, ML object, train samples, validation samples]
	"""
	
	frac, ml, train, traintruth, test, testturth, metric = params
	print "start", frac
	
	labelsdata = traintruth.reshape(len(traintruth), 1)
	traindata = np.concatenate((train, labelsdata), axis=1)
	ml.train(traindata)
	# We don't care about the uncertainty
	outcat, _ = ml.predict(train)
	
	testmetric = confusion.Confusion_Matrix(outcat, traintruth)
	train_err = np.mean(getattr(testmetric, metric)())

	outcat, _ = ml.predict(test)

	testmetric = confusion.Confusion_Matrix(outcat, testturth)
	test_err = np.mean(getattr(testmetric, metric)())

	print 'end', frac, train_err, test_err
	return frac, train_err, test_err


