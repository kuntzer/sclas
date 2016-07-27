import sclas
import os
import numpy as np
import pylab as plt

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)

demo = sclas.SClas('demo_ae')
datadir = "../../data/mnist"

#===================================================================================================
# Encoder definition and parameters
#===================================================================================================

encoder_params = {'rbm_type':'gd', 'architecture':[1000, 500, 200], 
				'layers_type':["SIGMOID",  "SIGMOID",  "SIGMOID", "LINEAR"]}
encoder_method = sclas.encoder.AutoEncoder(encoder_params)

#encoder_params = {'n_components':200}
#encoder_method = sclas.encoder.PCA(encoder_params)

#===================================================================================================
# Classifier definition and parameters
#===================================================================================================
#mlparams = sclas.classifier.MLParams(name = "foo", 
#		features = range(760),  labels = range(1))
#toolparams = sclas.classifier.fannwrapper.FANNParams(name = "bar", hidden_nodes = [20, 20, 20],
#        max_iterations = 2000, classifier="round", classifier_params={'nearest':1.})
#classifier_method = sclas.classifier.ML(mlparams, toolparams)
classifier_method = sclas.classifier.LogisiticRegression({'features':range(200), 'labels':range(1)})
#===================================================================================================
# Load the data
#===================================================================================================
traindata = sclas.utils.readpickle(os.path.join(datadir, "mnist-training-images.pkl"))
testdata = sclas.utils.readpickle(os.path.join(datadir, "mnist-testing-images.pkl"))
traindata = np.asarray(traindata, dtype=np.float) / 255.
testdata = np.asarray(testdata, dtype=np.float) / 255.

truth_train = sclas.utils.readpickle(os.path.join(datadir, "mnist-training-labels.pkl"))
truth_test = sclas.utils.readpickle(os.path.join(datadir, "mnist-testing-labels.pkl"))

#===================================================================================================
# Training
#===================================================================================================
demo.train_encoder(encoder_method, traindata, skipdone=False)
sclas.diagnostics.reconstruction(demo.encoder, testdata, truth_test)

demo.train_classifier(classifier_method, traindata, labelsdata=truth_train, skipdone=False, ncpu=1, n=1, strategy="one-vs-all")

#===================================================================================================
# Predicting which class is should be
#===================================================================================================
outcat, uncertainty = demo.predict(testdata)#, treat_catalog="one-vs-one")
if uncertainty is None:
	uncertainty = np.zeros_like(outcat)
"""
finalcat = np.zeros_like(truth_test)
print np.shape(finalcat)
for kk, l in enumerate(outcat.T): 
	if len(l[l > 0]) == 1:
		finalcat[kk] = l[l>0]
	elif len(l[l > 0]) > 1:
		finalcat[kk] = sclas.utils.around(np.median(l[l>0]), 0.5)

finalcat.reshape([len(finalcat), 1])

outcat = finalcat
"""

#===================================================================================================
# Now evaluate how good we did
#===================================================================================================

# This is using Testing
confusion = sclas.diagnostics.Confusion_Matrix(outcat, truth_test)
print 'f1 score', confusion.f1score().sum() / np.size(np.unique(truth_train))
plt.figure()
ax1=plt.subplot(111)
#xs = [confusion.accuracy(), confusion.precision(), confusion.recall(), confusion.f1score(), confusion.gmeasure()]
xs = [confusion.accuracy(), confusion.precision(), confusion.recall(), confusion.f1score(), confusion.gmeasure()]
labels = ['Accuracy', "Precision", "Recall", "F1-score", "G-measure"]
colors = ["red", "k", "b", "g", "yellow"]
sclas.plots.plot(ax1, xs, x_name="Stellar class", y_name="", labels=labels, colors=colors, 
				title='Diagnostics.Testing')


#===================================================================================================
# Show the result
#===================================================================================================

plt.figure()
ax0=plt.subplot(111)
comp = demo.encoder.encode(testdata)
sclas.plots.scatter(ax0, comp[:,0], comp[:,1], c=truth_test, x_name=r"c_1", y_name=r"c_2", 
	c_name="Spectral type", colorbarticks=range(10), 
	colorbarticklabels=range(10))


plt.figure()
ax=plt.subplot(111)
plt.plot([np.amin(truth_test),np.amax(truth_test)], [np.amin(truth_test),np.amax(truth_test)], 
		lw=2, color='red', ls='--')
sclas.plots.scatter(ax, outcat+uncertainty, truth_test, c=outcat, x_name="Classification", 
	y_name="Truth", c_name="Effective classification", colorbarticks=range(10), 
	colorbarticklabels=range(10))

plt.show()