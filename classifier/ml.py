"""
A module to connect the shape measurement stuff (astropy.table catalogs...) to the machine learning
(ml) wrappers.
It also takes care of masked columns, so that the underlying ml wrappers do not have to deal with
masked arrays.
"""

import numpy as np
from datetime import datetime
import os
from .. import method
import fannwrapper
import skynetwrapper

import copy

import logging
logger = logging.getLogger(__name__)




class MLParams:
	"""
	A container for the general parameters describing the machine learning:
	what features should I use to predict what labels ?
	"""
	
	def __init__(self, name, features, labels):
		"""
		Features, labels, and predlabels are lists of strings, with the names of the columns of
		the catalog to use.

		:param name: pick a short string describing these machine learning params.
			It will be used within filenames to store ML data.
		:param features: the list of column names to be used as input, i.e. "features", for
			the machine learning
		:param labels: the list of column names that the machine should learn to predict (that is, the "truth")
		
		To give a minimal example, to predict shear you could use something like::
			
			name = "simpleshear"
			features = ["mes_g1", "mes_g2", "mes_size", "mes_flux"]
			labels = ["tru_g1", "tru_g2"]
			predlabels = ["pre_g1", "pre_g2"]
		
		"""
		
		self.name = name
		self.features = features
		self.labels = labels
		

	def __str__(self):
		txt = "ML parameters \"%s\":\n" % (self.name)
		return txt


class ML(method.Method):
	"""
	This is a class that will hopefully nicely wrap any supervised machine learning regression code.
	
	This class has two key methods: train and predict. These methods call the "train"
	and "predict" methods of the underlying machine learning wrappers.
	For this to work, any machine learning wrapper has to implement methods with such names.
	"""

	def __init__(self, mlparams, toolparams, workbasedir = "."):
		"""
		:param mlparams: an MLParams object, describing *what* to learn.
		:param toolparams: this describes *how* to learn. For instance a FANNParams object, if you
			want to use FANN.
		:param workbasedir: path to a directory in which the machine parameters and any other
			intermediary or log files can be stored.
			Within this directroy, the training method will create a subdirectory using the
			mlparams.name and toolparams.name. So you can very well give the **same workbasedir**
			to different ML objects.
			If not specified (default), the current working directory will be used as workbasedir.
		"""
		
		self.mlparams = mlparams
		self.toolparams = toolparams
		self.workbasedir = workbasedir
		
		self.toolname = None # gets set below
		self.workdir = None # gets set below, depending on the tool to be used
		
		if isinstance(self.toolparams, fannwrapper.FANNParams):
			self.toolname = "FANN"
			self._set_workdir()
			self.tool = fannwrapper.FANNWrapper(self.toolparams, workdir=self.workdir)
		elif isinstance(self.toolparams, skynetwrapper.SkyNetParams):
			self.toolname = "SkyNet"
			self._set_workdir()
			self.tool = skynetwrapper.SkyNetWrapper(self.toolparams, workdir=self.workdir)
		else:
			raise RuntimeError("toolparams not recognized")
			# ["skynet", "ffnet", "pybrain", "fann", "minilut"]
		
	
	def __str__(self):
		"""
		A string describing an ML object, that will also be used as workdir name.
		"""
		return "ML_%s_%s_%s" % (self.toolname, self.mlparams.name, self.toolparams.name)
		
	
	def _set_workdir(self):
		self.workdir = os.path.join(self.workbasedir, str(self))
		
	def change_workdir(self, workdir):
		self.workdir = workdir
		self.tool.workdir = workdir
		
	def set_name(self, name):
		self.toolparams.name = name
	

	def get_workdir(self):
		logger.warning("This will be removed: directly use ml.workdir")
		return self.workdir
		
	
	def looks_same(self, other):
		"""
		Compares self to another ML object, and returns True if the objects seem to describe the "same"
		machine learning, in terms of parameters. Otherwise it returns False.
		
		:param other: an other ML object to compare with
		
		In principle this method might be called __eq__ to overwrite the default equality comparion,
		but given that it is not trivial it seems safer to just define it as a stand-alone function.
		"""
		return self.mlparams.__dict__ == other.mlparams.__dict__ and \
			self.toolparams.__dict__ == other.toolparams.__dict__ and \
			self.workbasedir == other.workbasedir and \
			self.toolname == other.toolname and \
			self.workdir == other.workdir


	def train(self, catalog):
		"""
		Runs the training, by extracting the numbers from the catalog and feeding them
		into the "train" method of the ml tool.
		
		:param catalog: an input astropy table. Has to contain the features and the labels.
			It can well be masked. Only those rows for which all the required
			features and labels are unmasked will be used for the training.
		
		"""	

		starttime = datetime.now()

		# make sure the workdir is correct:
		self.tool.workdir = self.workdir
			
		logger.info("Training %s  in %s..." % (str(self), self.workdir))
		logger.info(str(self.mlparams))
		logger.info(str(self.toolparams))
				
		# Now we turn the relevant rows and columns of the catalog into unmasked 2D numpy arrays.
		# There are several ways of turning astropy.tables into plain numpy arrays.
		# The shortest is np.array(table...)
		# I use the following one, as I find it the most explicit (in terms of respecting
		# the order of features and labels).
		# We could add dtype control, but the automatic way should work fine.

		featuresdata = catalog[:,self.mlparams.features]	
		idlabel = np.array(self.mlparams.labels) + \
			self.mlparams.features[-1] + \
			1
		labelsdata = catalog[:,idlabel]

		# And we call the tool's train method:
		self.tool.train(features=featuresdata, labels=labelsdata)
		
		endtime = datetime.now()
		logger.info("Done! This training took %s" % (str(endtime - starttime)))
		


	def predict(self, catalog):
		"""
		Computes the prediction(s) based on the features in the given catalog.
		Of course it will return a new astropy.table to which the new "predlabels" columns are added,
		instead of changing your catalog in place.
		If any feature values of your catalog are masked, the corresponding rows will not be predicted,
		and the predlabels columns will get masked accordingly.
		
		"""

		# We can run the prediction tool, it doesn't have to worry about any masks:
		preddata = self.tool.predict(features=catalog)
		

		return preddata
	
		

