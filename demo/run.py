import sclas
import os
import numpy as np
import pylab as plt

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)

stellar_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
all_stellar_types = ['%s%s' % (letter, ii * 5) for letter in stellar_types for ii in range(2)]
all_stellar_types += ['AGB', 'WD']


demo = sclas.SClas('demo')

#===================================================================================================
# Encoder definition and parameters
#===================================================================================================
encoder_params = {'n_components':2}
encoder_method = sclas.encoder.PCA(encoder_params)

#===================================================================================================
# Classifier definition and parameters
#===================================================================================================
mlparams = sclas.classifier.MLParams(name = "foo", 
		features = range(encoder_params['n_components']),  labels = range(1))
toolparams = sclas.classifier.fannwrapper.FANNParams(name = "bar", hidden_nodes = [20, 20, 20],
        max_iterations = 1000, classifier="round", classifier_params={'nearest':.5})
classifier_method = sclas.classifier.ML(mlparams, toolparams)

#===================================================================================================
# Load the data
#===================================================================================================
datapklfname = os.path.join("../../data/multipsf", "train_data.pkl")
traindata, noise_maps = sclas.utils.readpickle(datapklfname)
datapklfname = os.path.join("../../data/multipsf", "test_data.pkl")
testdata, noise_maps = sclas.utils.readpickle(datapklfname)

truth_train = np.loadtxt("../../data/multipsf/star_truth.dat")[:2500,1]
truth_test = np.loadtxt("../../data/multipsf/star_truth.dat")[2500:,1]

#===================================================================================================
# Training
#===================================================================================================
demo.train_encoder(encoder_method, traindata, skipdone=True);
demo.train_classifier(classifier_method, traindata, labelsdata=truth_train, skipdone=True)

#===================================================================================================
# Predicting which class is should be
#===================================================================================================
outcat, uncertainty = demo.predict(testdata)

#===================================================================================================
# Now evaluate how good we did
#===================================================================================================

# This is using Testing
confusion = sclas.diagnostics.Confusion_Matrix(outcat, truth_test)
print 'f1 score', confusion.f1score().sum()
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
	c_name="Spectral type", colorbarticks=np.arange(2, len(all_stellar_types))/2., 
	colorbarticklabels=all_stellar_types)


plt.figure()
ax=plt.subplot(111)
plt.plot([np.amin(truth_test),np.amax(truth_test)], [np.amin(truth_test),np.amax(truth_test)], 
		lw=2, color='red', ls='--')
sclas.plots.scatter(ax, outcat+uncertainty, truth_test, c=outcat, x_name="Classification", 
	y_name="Truth", c_name="Effective classification", colorbarticks=np.arange(2, len(all_stellar_types))/2., 
	colorbarticklabels=all_stellar_types)

plt.show()