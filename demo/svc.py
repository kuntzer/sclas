import sclas
import os
import numpy as np
import pylab as plt

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)

stellar_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
all_stellar_types = ['%s%s' % (letter, ii * 5) for letter in stellar_types for ii in range(2)]
all_stellar_types += ['AGB', 'WD']
all_stellar_types_names = [r"$\mathrm{%s}$" % (cl) for cl in all_stellar_types]


demo = sclas.SClas('demo_multipsf_svc')
datadir = "../../data/multipsf_noisechange"
datadir_test = "../../data/600psf_noisechange"
Lambda = 1

# Do you want to save the figures to the disk ?
save_figures = False
# Do you want to see the figures outputed to the display ?
show = True

#===================================================================================================
# Encoder definition and parameters
#===================================================================================================
encoder_params = {'n_components':5}
encoder_method = sclas.encoder.PCA(encoder_params)

#===================================================================================================
# Classifier definition and parameters
#===================================================================================================
"""
mlparams = sclas.classifier.MLParams(name = "foo", 
		features = range(encoder_params['n_components']),  labels = range(1))
toolparams = sclas.classifier.fannwrapper.FANNParams(name = "bar", hidden_nodes = [20, 20, 20],
        max_iterations = 2000, classifier="round", classifier_params={'nearest':.5})
classifier_method = sclas.classifier.ML(mlparams, toolparams)
"""
classifier_params = {'features': range(encoder_params['n_components']), 'labels': range(1), 'gamma': 150}
classifier_method = sclas.classifier.SupportVectorClassifier(classifier_params)

#===================================================================================================
# Load the data
#===================================================================================================
datapklfname = os.path.join(datadir, "train_data_lambda_%d.pkl" % Lambda)
traindata, noise_maps = sclas.utils.readpickle(datapklfname)
datapklfname = os.path.join(datadir_test, "test_data_lambda_%d.pkl" % Lambda)
testdata, noise_maps = sclas.utils.readpickle(datapklfname)

truth_train = np.loadtxt(os.path.join(datadir, "star_truth.dat"))[:2500,1]
truth_test = np.loadtxt(os.path.join(datadir_test, "star_truth.dat"))[2500:,1]
#===================================================================================================
# Training
#===================================================================================================
demo.train_encoder(encoder_method, traindata, skipdone=True)
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
print 'f1 score', confusion.f1score().sum() / np.size(np.unique(truth_train))
sclas.plots.figures.set_fancy()

f1 = plt.figure()
ax1 = plt.subplot(111)
xs = [confusion.accuracy(), confusion.precision(), confusion.recall(), confusion.f1score(), confusion.gmeasure()]
labels = ['Accuracy', "Precision", "Recall", "F1-score", "G-measure"]
labels = ["$\mathrm{%s}$" % l for l in labels]
colors = ["red", "k", "b", "g", "yellow"]
sclas.plots.plot(ax1, xs, x_name=r"$\mathrm{Spectral class}$", y_name="", labels=labels, colors=colors, 
				title=r'$\mathrm{Confusion Matrix}$')
if save_figures:
	sclas.plots.figures.savefig(os.path.join(demo.workdir, "figures", "confusion"), f1, fancy=True)

#===================================================================================================
# Show the result
#===================================================================================================

f2 = plt.figure()
ax0 = plt.subplot(111)
comp = demo.encoder.encode(testdata)
sclas.plots.scatter(ax0, comp[:,0], comp[:,1], c=truth_test, x_name=r"$\mathrm{Component\ 1}$", 
	y_name=r"$\mathrm{Component\ 2}$", c_name=r"$\mathrm{Spectral\ type}$", 
	colorbarticks=np.arange(2, len(all_stellar_types))/2., colorbarticklabels=all_stellar_types_names)
if save_figures:
	sclas.plots.figures.savefig(os.path.join(demo.workdir, "figures",  "first2components"), f2, fancy=True)

f = plt.figure()
ax=plt.subplot(111)

plt.plot([np.amin(truth_test),np.amax(truth_test)], [np.amin(truth_test),np.amax(truth_test)], 
		lw=2, color='red', ls='--')
sclas.plots.scatter(ax, outcat+uncertainty, truth_test, c=outcat.reshape(len(outcat)), x_name=r"$\mathrm{Classification}$", 
	y_name=r"$\mathrm{Truth}$", c_name=r"$\mathrm{Effective\ classification}$", 
	colorbarticks=np.arange(2, len(all_stellar_types))/2., 	colorbarticklabels=all_stellar_types_names)
if save_figures:
	sclas.plots.figures.savefig(os.path.join(demo.workdir, "figures",  "classification"), f, fancy=True)
	
f3 = plt.figure()
ax=plt.subplot(111)

sclas.plots.matshow(ax, confusion.matrix, 
				x_name=r"$\mathrm{Classification}$", y_name=r"$\mathrm{Truth}$", 
				x_n=confusion.n_classes, y_n=confusion.n_classes, 
		x_tickslabels=all_stellar_types_names, y_tickslabels=all_stellar_types_names)
print confusion.matrix[::-1,:]

ax.grid(True)
if save_figures:
	sclas.plots.figures.savefig(os.path.join(demo.workdir, "figures",  "confusion_matrix"), f3, fancy=True)
	
if show:
	plt.show()