import sclas
import os
import numpy as np
import pylab as plt
from sklearn import datasets

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)

demo = sclas.SClas('demo_digits_svc')

# Do you want to save the figures to the disk ?
save_figures = False
# Do you want to see the figures outputed to the display ?
show = True

#===================================================================================================
# Encoder definition and parameters
#===================================================================================================
encoder_params = {'n_components':32}
encoder_method = sclas.encoder.PCA(encoder_params)

#===================================================================================================
# Classifier definition and parameters
#===================================================================================================
classifier_params = {'features': range(encoder_params['n_components']), 'labels': range(1), 'gamma': 0.0025}
classifier_method = sclas.classifier.SupportVectorClassifier(classifier_params)

#===================================================================================================
# Load the data
#===================================================================================================
digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))

# To apply sclas on this data, we need to flatten the image, to
# turn the data in (samples), (labels) vectors:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
ns = 950
traindata = data[:ns]
truth_train = digits.target[:ns]
testdata =  data[ns:]
truth_test = digits.target[ns:]
#===================================================================================================
# Training
#===================================================================================================
demo.train_encoder(encoder_method, traindata, skipdone=False)
demo.train_classifier(classifier_method, traindata, labelsdata=truth_train, skipdone=False)

#===================================================================================================
# Predicting which class is should be
#===================================================================================================
outcat_train, uncertainty_train = demo.predict(traindata)
outcat, uncertainty = demo.predict(testdata)

#===================================================================================================
# Now evaluate how good we did
#===================================================================================================

confusion = sclas.diagnostics.Confusion_Matrix(outcat_train, truth_train)
f1 = confusion.f1score().sum() / np.size(np.unique(truth_train))
print 'f1 score on train %1.3f (%1.2f%% error)' % (f1, 100.*(1.-f1)) 

# This is using Testing
confusion = sclas.diagnostics.Confusion_Matrix(outcat, truth_test)
f1 = confusion.f1score().sum() / np.size(np.unique(truth_train))
print 'f1 score on test %1.3f (%1.2f%% error)' % (f1, 100.*(1.-f1)) 
sclas.plots.figures.set_fancy()

f1 = plt.figure()
ax1 = plt.subplot(111)
xs = [confusion.accuracy(), confusion.precision(), confusion.recall(), confusion.f1score(), confusion.gmeasure()]
labels = ['Accuracy', "Precision", "Recall", "F1-score", "G-measure"]
colors = ["red", "k", "b", "g", "yellow"]
sclas.plots.plot(ax1, xs, x_name="Spectral class", y_name="", labels=labels, colors=colors, 
				title='Confusion Matrix')
if save_figures:
	sclas.plots.figures.savefig(os.path.join(demo.workdir, "figures", "confusion"), f1, fancy=True)

#===================================================================================================
# Show the result
#===================================================================================================

f0 = plt.figure()
ax0 = plt.subplot(111)
comp = demo.encoder.encode(testdata)
sclas.plots.scatter(ax0, comp[:,0], comp[:,1], c=truth_test, x_name=r"Component 1", 
	y_name=r"Component 2", c_name="Digit", 
	colorbarticks=np.arange(10), colorbarticklabels=np.arange(10))
if save_figures:
	sclas.plots.figures.savefig(os.path.join(demo.workdir, "figures",  "first2components"), f0, fancy=True)

f = plt.figure()
ax=plt.subplot(111)

plt.plot([np.amin(truth_test),np.amax(truth_test)], [np.amin(truth_test),np.amax(truth_test)], 
		lw=2, color='red', ls='--')

sclas.plots.scatter(ax, outcat+uncertainty, truth_test, c=truth_test, x_name="Classification", 
	y_name="Truth", c_name="Effective classification", 
	colorbarticks=np.arange(10), colorbarticklabels=np.arange(10))
if save_figures:
	sclas.plots.figures.savefig(os.path.join(demo.workdir, "figures",  "classification"), f, fancy=True)
	
f3 = plt.figure()
ax=plt.subplot(111)

sclas.plots.matshow(ax, confusion.matrix, 
				x_name=r"$\mathrm{Classification}$", y_name=r"$\mathrm{Truth}$", 
				x_n=confusion.n_classes, y_n=confusion.n_classes, 
		x_tickslabels=np.arange(10), y_tickslabels=np.arange(10))
print confusion.matrix[::-1,:]

ax.grid(True)
if save_figures:
	sclas.plots.figures.savefig(os.path.join(demo.workdir, "figures",  "confusion_matrix"), f3, fancy=True)


if show:
	plt.show()