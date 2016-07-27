"""
Wrapper around SkyNet neural network regression
http://arxiv.org/abs/1309.0790
"""

import csv
import os
import numpy as np
from datetime import datetime
import copy

import logging
logger = logging.getLogger(__name__)


"""
Some helper functions
"""

def readdata(filename):
	"""
	Reads training data files, low level function
	"""
	f = open(filename, 'rb')
	reader = csv.reader(f, delimiter=',')
	nin = int(reader.next()[0])
	nout = int(reader.next()[0])
	inputs = []
	outputs = []
	for row in reader:
		inputs.append(map(float, row[:-1]))
		outputs.append(map(float, reader.next()[:-1]))
	f.close()
	assert np.alltrue(np.array(map(len, inputs)) == nin)
	#assert np.alltrue(np.array(map(len, outputs)) == nout)
	inputs = np.array(inputs)
	outputs = np.array(outputs)
	assert len(inputs) == len(outputs)
	ndp = len(inputs)
	print "Read %s\n(datapoints : %i, inputs : %i, outputs : %i)" % (filename, ndp, nin, nout)
	
	return (inputs, outputs)



def writedata(filename, inputs, outputs=None, noutput=1):
	"""
	Writes data files to be read by SkyNet, low level function
	
	If you do not specify outputs (e.g., for predictions), I will write noutput zeroes.
	Having the correct number of output labels seems mandatory for SkyNet.
	
	"""
	if outputs == None:
		outputs = np.zeros(shape=(np.shape(inputs)[0], noutput))
	assert np.shape(inputs)[0] == np.shape(outputs)[0]
	ndp = np.shape(inputs)[0]
	nin = np.shape(inputs)[1]
	nout = np.shape(outputs)[1]
	f = open(filename, "w")
	f.write("%i,\n" % nin)
	f.write("%i,\n" % nout)
	for inputline, outputline in zip(inputs, outputs):
		f.write(",".join(["%.6f" % item for item in inputline]) + ",\n")
		f.write(",".join(["%.6f" % item for item in outputline]) + ",\n")
	f.close()
	print "Wrote %s\n(datapoints : %i, inputs : %i, outputs : %i)" % (filename, ndp, nin, nout)


def readpred(filename):
	"""
	Read out the "raw" output
	"""
	data = np.loadtxt(filename)
	return data


	
class SkyNetParams:
	"""
	A container for the internal parameters for SkyNet
	
	"""
	
	def __init__(self, hidden_nodes, pretrain=0, sigma=0.1, confidence_rate=0.3,
		     confidence_rate_minimum = 0.01, iteration_print_frequency = 10,
		     max_iterations = 200, name="default", classifier="None", classifier_params=None):
		"""
		:param hidden_nodes: a list of number of nodes in the hidden layers.
		                     example: (5) means one hidden layer with 5 nodes...
		"""
		self.hidden_nodes = hidden_nodes
		self.pretrain = pretrain
		self.sigma = sigma
		self.confidence_rate = confidence_rate
		self.confidence_rate_minimum = confidence_rate_minimum
		self.iteration_print_frequency = iteration_print_frequency
		self.max_iterations = max_iterations
		self.name = name # Just a string to identify this parameter set...
		self.classifier = classifier
		self.classifier_params = classifier_params


class SkyNetWrapper():
	"""
	Represents a SkyNet run
	"""
	
	def __init__(self, params, workdir = None):
		"""
		params is a SkyNetParams object
		"""
		
		if workdir == None:
			self.workdir = "SkyNet_workdir"
		else:
			self.workdir = workdir
		
		self.params = params
		


	def __str__(self):
		return "SkyNet in %s: max_iterations %i" % (os.path.basename(self.workdir),
							    self.params.max_iterations)


	def _prep(self, features, labels):
		"""
		Prepare a SkyNet training, by writing the input files. This is called by train.
		features is a 2D numpy array, each line (first index) contains the features of
		one element.
		labels contains the corresponding labels.
		Those two must have the same number of lines
		
		"""

		assert np.shape(features)[0] == np.shape(labels)[0]
		
		print "Preparing SkyNet run on %i elements in %s..." % (np.shape(features)[0],
									self.workdir)

		if not os.path.isdir(self.workdir):
			os.makedirs(self.workdir)
		else:
			logger.warning("SkyNet workdir %s already exists, I will overwrite stuff !" % \
				self.workdir)
				
		# How many validation points to use:
		cut = int(float(np.shape(features)[0])/4.0)
		
		# We shuffle the features and corresponding labels
		indexes = np.arange(0, np.shape(features)[0])
		np.random.shuffle(indexes)
		
		trainindexes = indexes[cut:]
		testindexes = indexes[:cut]
		
		trainfeatures = features[trainindexes,:]
		testfeatures = features[testindexes,:]
		
		trainlabels = labels[trainindexes,:]
		testlabels = labels[testindexes,:]
		
		self.nin = np.shape(features)[1] # number of features per point
		self.nout = np.shape(labels)[1]
		
		writedata(os.path.join(self.workdir, "skynet_train.txt"), trainfeatures, trainlabels)
		writedata(os.path.join(self.workdir, "skynet_test.txt"), testfeatures, testlabels)

		self.nhidbloc = "\n".join(["#nhid\n%i" % (e) for e in self.params.hidden_nodes])
		
		skynetinptxt = """#input_root
skynet_
#output_root
skynet_
#verbose
3
#pretrain
0
#validation_data
1
{self.nhidbloc}
#classification_network
0
#mini-batch_fraction
1
#prior
1
#whitenin
1
#whitenout
1
#noise_scaling
1
#set_whitened_noise
1
#sigma
{self.params.sigma}
#confidence_rate
{self.params.confidence_rate}
#confidence_rate_minimum
{self.params.confidence_rate_minimum}
#iteration_print_frequency
{self.params.iteration_print_frequency}
#fix_seed
0
#fixed_seed
1
#calculate_evidence
1
#resume
0
#historic_maxent
0
#recurrent
0
#convergence_function
1
#max_iter
{self.params.max_iterations}
""".format(self=self)
	

		# Absolute paths don't work, filesnames too long ?

		skynetinp = open(os.path.join(self.workdir, "skynet.inp"), "w")
		skynetinp.write(skynetinptxt)
		skynetinp.close()

	
	def train(self, features, labels, exe = "nice -n 15 SkyNet", verbose=False):
		"""
		Top-level function to train a SkyNet network.
		"""
		# First we prepare the input files:
		self._prep(features, labels)
		
		print "Starting the training of"
		print str(self)
		
		starttime = datetime.now()
		
		origdir = os.getcwd()
		os.chdir(self.workdir)
	
		#os.system("nice -n 19 SkyNet " + os.path.join(workdir, "skynet.inp"))

		#os.system("SkyNet skynet.inp")
		#os.system("nice -n 19 SkyNet skynet.inp")
		os.system("%s skynet.inp" % (exe))
	
		os.chdir(origdir)
		endtime = datetime.now()
		if verbose:
			print "This training took %s" % (str(endtime - starttime))
	
	
	def predict(self, features, exe = "nice -n 15 CalPred"):
		"""
		Top-level function to compute predictions.
		Give me features, I return the corresponding labels
		"""
	
		# The filepaths
		in_net = os.path.join(self.workdir, "skynet_network.txt")
		in_train = os.path.join(self.workdir, "skynet_train.txt")
		in_predsim = os.path.join(self.workdir, "skynet_pred_in.txt")
		out_predsim = os.path.join(self.workdir, "skynet_pred_out.txt")

		print "Writing CalPred input and running it..."

		writedata(in_predsim, features, noutput = self.nout)

		# The options for CalPred :
		"""
		1) data type: 0=blind, 1=known outputs
		2) network type: 0=reg, 1=class, 2=text, 3=autoencoder
		3) network type: 0=reg, 1=recurrent
		4) input network save file
		5) input training data file
		6) data set file
		7) output file
		8) loglike function: 0=none, 1=standard, 2=HHPScore, 3=standard on log(out+1)
		9) print error? 0=no, 1=yes
		10) read in accuracies? 0=no, 1=yes
		11) output transformation file (optional)
		"""

		cmd = "%s 0 0 0 %s %s %s %s 0 0 0" % (exe, in_net, in_train, in_predsim, out_predsim)
		os.system(cmd)
		
		print "Reading CalPred output..."

		pred = readpred(out_predsim)

		assert pred.shape[0] == np.shape(features)[0]
		assert pred.shape[1] == self.nin + 2*self.nout

		pred = pred[:,-self.nout:]
		
		if self.params.classifier == "round":
			clp = self.params.classifier_params['nearest']
			output_classified = np.round(pred / clp) * clp
			output_uncertainty = (pred - output_classified) #/ clp
		elif self.params.classifier == "None":
			pred = np.asarray(pred)
			output_classified = copy.copy(pred)
			output_uncertainty = 0.
		else:
			raise ValueError("SkyNet classifier '%s' type unknown" % self.params.classifier)
		

		return output_classified, output_uncertainty


if __name__ == "__main__":
	"""
	A little demo and test !
	"""
	import matplotlib.pyplot as plt
	import logging
	logging.basicConfig(level=logging.DEBUG)
	
	def ytrue(x):
		return np.sin(x)*x

	# Simple 1D case : we predict y(x) from noisy data points.
	x = np.random.uniform(0, 10, 100)
	y = ytrue(x) + 0.6*np.random.randn(len(x))
	
	# x and y are 1D arrays. But our SkyNet wrapper works only with 2D arrays.
	# First index = datapoint, second index : the different features.
	
	features = x.reshape(100, 1)
	labels = y.reshape(100, 1)
	
	snparams = SkyNetParams(hidden_nodes = [5], max_iterations=100)
	
	sn = SkyNetWrapper(snparams)
	sn.train(features, labels)
	
	# Let's make some predictions on a fine grid of points :
	pred_features = np.linspace(-2, 12, 1000).reshape(1000, 1)
	pred_labels, _ = sn.predict(pred_features)
	
	x1 = np.linspace(-2,12)
	
	plt.plot(x, y, "r.")
	plt.plot(pred_features, pred_labels, "g-")
	plt.plot(x1, ytrue(x1), "k--")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.show()

	"""
	# More sophisticated 2D input and output :
	x = np.random.uniform(0, 10, (100, 2))
	
	# Two surfaces
	y1 = x[:,0] + x[:,1]# + np.random.randn(len(x))
	y2 = x[:,0] * x[:,1]
	y = np.vstack((y1, y2)).T
	
	snparams = SkyNetParams(hidden_nodes = [5], max_iterations=100)
	sn = SkyNetWrapper(snparams)

	sn.train(x, y)
	
	pred_y, _ = sn.predict(x)
	
	plt.subplot(1, 2, 1)
	plt.scatter(y[:,0], pred_y[:,0])
	plt.xlabel("true y1")
	plt.ylabel("predicted y1")
	plt.subplot(1, 2, 2)
	plt.scatter(y[:,1], pred_y[:,1])
	plt.xlabel("true y2")
	plt.ylabel("predicted y2")
	
	plt.show()
	"""
	
	
	



