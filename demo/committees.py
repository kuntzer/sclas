import sclas
import os
import numpy as np
import pylab as plt

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)

stellar_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
all_stellar_types = ['%s%s' % (letter, ii * 5) for letter in stellar_types for ii in range(2)]
all_stellar_types += ['AGB', 'WD']
all_stellar_types = all_stellar_types[1:]
all_stellar_types_names = [r"$\mathrm{%s}$" % (cl) for cl in all_stellar_types]
"""
stellar_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
all_stellar_types = ['%s' % (letter) for letter in stellar_types for ii in range(1)]
#all_stellar_types += ['AGB', 'WD']
all_stellar_types_names = [r"$\mathrm{%s}$" % (cl) for cl in all_stellar_types]
"""
scl = sclas.SClas('demo_committes_HST814_step_XL')

datadir = "../../data/multipsf_noisechange"
datadir_test = "../../data/600psf_noisechange"
"""
datadir = "../../data/spectra_HST_F814W_81_10psf"
datadir_test = "../../data/spectra_HST_F814W_81_600psf"

datadir = "../../data/spectra_10psf_HST"
datadir_test = "../../data/spectra_600psf_HST"

datadir = "../../data/spectra_HST_F814W_step_10psf"
datadir_test = "../../data/spectra_HST_F814W_step_600psf"
"""
#datadir = "../../data/spectra_HST_F814W_stepoldfashioned_10psf"
#datadir_test = "../../data/spectra_HST_F814W_stepoldfashioned_600psf"

#datadir = "../../data/spectra_HST_F606W_step_10psf"
#datadir_test = "../../data/spectra_HST_F606W_step_600psf"

#datadir = "../../data/spectra_test_F606step"
#datadir_test = "../../data/spectra_test_F606step"

Lambda = 1

# Do you want to save the figures to the disk ?
save_figures = False

# Do you want to see the figures outputed to the display ?
show = True

##import matplotlib.font_manager as fm
##from matplotlib import rc
##prop = fm.FontProperties(fname='/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreadventor-regular.otf')
#rc('font', **{'fname':'/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreadventor-regular.otf'})
##rc('font', **{'family':'TeX Gyre Adventor','size':14})
#===================================================================================================
# Encoder definition and parameters
#===================================================================================================
encoder_params = {'n_components': 8}
encoder_method = sclas.encoder.PCA(encoder_params)

#===================================================================================================
# Classifier definition and parameters
#==================================================================================================="""

mlparams = sclas.classifier.MLParams(name = "FANN", 
		features = range(encoder_params['n_components']),  labels = range(1))
toolparams = sclas.classifier.fannwrapper.FANNParams(name = "bar", hidden_nodes = [20, 20, 20],
        max_iterations = 2000, classifier="round", classifier_params={'nearest':0.5})
classifier_method = sclas.classifier.ML(mlparams, toolparams)
#===================================================================================================
# Load the data
#===================================================================================================
datapklfname = os.path.join(datadir, "train_data_lambda_%g.pkl" % Lambda)
traindata, noise_maps = sclas.utils.readpickle(datapklfname)
"""
datapklfname = os.path.join(datadir, "test_data_lambda_%g.pkl" % Lambda)
traindata, noise_maps = sclas.utils.readpickle(datapklfname)

traindata = np.vstack([traindata, traindata2])

print np.shape(traindata)
exit()"""
#traindata = traindata[:2500,:]

datapklfname = os.path.join(datadir_test, "test_data_lambda_%g.pkl" % Lambda)
testdata, noise_maps = sclas.utils.readpickle(datapklfname)
truth_train = np.loadtxt(os.path.join(datadir, "star_truth.dat"))[:2500,1]#[:1000,1]#
truth_test = np.loadtxt(os.path.join(datadir_test, "star_truth.dat"))[2500:,1]#[1000:3000,1]#

#truth_test = np.floor(truth_test)
#truth_train = np.floor(truth_train)
#===================================================================================================
# Training
#===================================================================================================
"""
for ii in range(10):
	plt.figure()
	plt.imshow(np.log10(traindata[ii].reshape(42,42)), interpolation="nearest")
	plt.title(truth_train[ii])

plt.show()
exit()"""

scl.train_encoder(encoder_method, traindata, skipdone=True)
scl.train_classifier(classifier_method, traindata, truth_train, n=8, ncpu=8, skipdone=True)

#===================================================================================================
# Testing the versability of the training
#===================================================================================================
for tort, data, truth in zip(["training", "test"], [traindata, testdata],[truth_train, truth_test]):
	print '**** %s *****' % (tort)
	catalogs, _ = scl.predict(data)
	random = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7., 7.5], size=np.shape(catalogs[0]))
	nclass = np.size(np.unique(truth_train))
	#random = np.random.choice([1, 2, 3, 4, 5, 6, 7.], size=np.shape(catalogs[0]))
	print 'Random f1', sclas.diagnostics.Confusion_Matrix(random, truth).f1score().sum() / nclass 

	if tort == "training":
		scores = []
		highest = 0
		id_hi = 0
		for ii, committee_member in enumerate(catalogs):
			scores.append(sclas.diagnostics.Confusion_Matrix(sclas.utils.around( \
				committee_member, 0.5, amin=1, amax=7.5), truth).f1score())
			if np.sum(scores[-1]) > highest:
				highest = np.sum(scores[-1]) 
				id_hi = ii
		scores = np.asarray(scores)**2
		scores_per_member = np.sqrt(scores).sum(axis=1) / nclass
		member_keep = 8
		ids_members = np.argpartition(scores_per_member, -member_keep)[-member_keep:]
		ids_members = np.random.randint(0, 8, member_keep)
	
	
	weights = np.sum(scores, axis=1)
	
	wavg = np.average(catalogs[ids_members,:], weights=weights[ids_members], axis=0)
	medi = np.median(catalogs[ids_members,:], axis=0)
	mea = np.mean(catalogs[ids_members,:], axis=0)
	uncertainty = np.std(catalogs, axis=0)
	
	
	wavg = sclas.utils.around(wavg, 0.5, amin=1, amax=7.5)
	medi = sclas.utils.around(medi, 0.5, amin=1, amax=7.5)
	mea = sclas.utils.around(mea, 0.5, amin=1, amax=7.5)
	f1wavg = sclas.diagnostics.Confusion_Matrix(wavg, truth).f1score()
	f1medi = sclas.diagnostics.Confusion_Matrix(medi, truth).f1score()
	f1mea = sclas.diagnostics.Confusion_Matrix(mea, truth).f1score()
	
	nclass = np.size(np.unique(truth_train))
	
	print 'w.  avg\tFailures:', np.size(np.nonzero(wavg - truth))/float(len(truth)) * 100, \
		'%\t F1 = ', f1wavg.sum() / nclass
	print 'Median\tFailures:', np.size(np.nonzero(medi - truth))/float(len(truth)) * 100, \
		'%\t F1 = ', f1medi.sum() / nclass
	print 'mean\tFailures:', np.size(np.nonzero(mea - truth))/float(len(truth)) * 100, \
		'%\t F1 = ', f1mea.sum() / nclass
	print np.mean(uncertainty), '+/-', np.std(uncertainty)/np.shape(catalogs)[0]
	outcat = wavg

sclas.utils.writepickle(f1medi, os.path.join(scl.workdir, "data", "f1score_lambda%d.pkl" % Lambda))
	
confusion = sclas.diagnostics.Confusion_Matrix(outcat, truth_test)
#sclas.plots.figures.set_fancy()

f1 = plt.figure()
ax1 = plt.subplot(111)
xs = [confusion.accuracy(), confusion.precision(), confusion.recall(), confusion.f1score(), confusion.gmeasure()]
labels = ['Accuracy', "Precision", "Recall", "F1-score", "G-measure"]
labels = ["$\mathrm{%s}$" % l for l in labels]
colors = ["red", "k", "b", "g", "yellow"]
sclas.plots.plot(ax1, xs, x_name=r"$\mathrm{Spectral class}$", y_name="", labels=labels, colors=colors, 
				title=r"$\mathrm{Confusion Matrix}$")
if save_figures:
	sclas.plots.figures.savefig(os.path.join(scl.workdir, "figures", "confusion"), f1, fancy=True)

import subprocess
fname = os.path.join(scl.workdir, "figures", "confusion_hst")
f1.savefig(fname+'.png',dpi=300)
f1.savefig(fname+'.pdf',transparent=True)
#os.system("epstopdf "+fname+".eps")
command = 'pdfcrop %s.pdf' % fname
subprocess.check_output(command, shell=True)
os.system('mv '+fname+'-crop.pdf '+fname+'.pdf')
#===================================================================================================
# Show the result
#===================================================================================================
print all_stellar_types; 
print all_stellar_types_names
print np.unique(truth_test)

f0 = plt.figure()
ax0 = plt.subplot(111)
comp = scl.encoder.encode(testdata)
sclas.plots.scatter(ax0, comp[:,0], comp[:,1], c=truth_test, x_name=r"$\mathrm{Component\ 1}$", 
	y_name=r"$\mathrm{Component\ 2}$", c_name=r"$\mathrm{Spectral\ type}$", title=scl.encoder,
	colorbarticks=np.arange(2, len(all_stellar_types[1:]))/2., 
	colorbarticklabels=all_stellar_types_names[1:], cmap=plt.cm.get_cmap('winter'))
if save_figures:
	sclas.plots.figures.savefig(os.path.join(scl.workdir, "figures",  "first2components"), f0, fancy=True)

fname = os.path.join(scl.workdir, "figures", "first2components")
f0.savefig(fname+'.png',dpi=300)
f0.savefig(fname+'.pdf',transparent=True)
#os.system("epstopdf "+fname+".eps")
command = 'pdfcrop %s.pdf' % fname
subprocess.check_output(command, shell=True)
os.system('mv '+fname+'-crop.pdf '+fname+'.pdf')

f = plt.figure()
ax=plt.subplot(111)
plt.plot([np.amin(truth_test),np.amax(truth_test)], [np.amin(truth_test),np.amax(truth_test)], 
		lw=2, color='red', ls='--')
sclas.plots.scatter(ax, outcat+uncertainty, truth_test, c=outcat, x_name=r"$\mathrm{Classification}$", 
	y_name=r"$\mathrm{Truth}$", c_name=r"$\mathrm{Effective classification}$", colorbarticks=np.arange(2, len(all_stellar_types))/2., 
	colorbarticklabels=all_stellar_types)
if save_figures:
	sclas.plots.figures.savefig(os.path.join(scl.workdir, "figures",  "classification"), f, fancy=True)
	
f3 = plt.figure()
ax=plt.subplot(111)

sclas.plots.matshow(ax, confusion.matrix, 
				x_name=r"$\mathrm{Classification}$", y_name=r"$\mathrm{Truth}$", 
				x_n=confusion.n_classes, y_n=confusion.n_classes, 
		x_tickslabels=all_stellar_types_names, y_tickslabels=all_stellar_types_names)
print confusion.matrix[::-1,:]

ax.grid(True)
if save_figures:
	sclas.plots.figures.savefig(os.path.join(scl.workdir, "figures",  "confusion_matrix"), f3, fancy=True)

	
if show:
	plt.show()