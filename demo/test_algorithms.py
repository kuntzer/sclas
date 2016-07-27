import sclas
import os
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)

stellar_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
all_stellar_types = ['%s%s' % (letter, ii * 5) for letter in stellar_types for ii in range(2)]
all_stellar_types += ['AGB', 'WD']
all_stellar_types_names = [r"$\mathrm{%s}$" % (cl) for cl in all_stellar_types]


datadir = "../../data/multipsf_noisechange"
datadir_test = "../../data/600psf_noisechange"

Lambda = 1

outfoler = 'data'
outfile = "multipsf_noisechange.dat"

#===================================================================================================
# Load the data
#===================================================================================================
datapklfname = os.path.join(datadir, "train_data_lambda_%d.pkl" % Lambda)
traindata, noise_maps = sclas.utils.readpickle(datapklfname)
datapklfname = os.path.join(datadir_test, "test_data_lambda_%d.pkl" % Lambda)
testdata, noise_maps = sclas.utils.readpickle(datapklfname)
truth_train = np.loadtxt(os.path.join(datadir, "star_truth.dat"))[:2500,1]
truth_test = np.loadtxt(os.path.join(datadir_test, "star_truth.dat"))[2500:,1]

for n_components in [5, 10]:#, 20, 30]:
#===================================================================================================
# Encoder definition and parameters
#===================================================================================================
	encoders = [
		sclas.encoder.PCA({'n_components': n_components}),
		#sclas.encoder.ICA({'n_components': n_components}),	
		#sclas.encoder.NMF({'n_components': n_components}),
		#sclas.encoder.LLE({'n_components': n_components}),
			]

#===================================================================================================
# Classifier definition and parameters
#===================================================================================================
	mlparams = sclas.classifier.MLParams(name = "FANN", features = range(n_components),  labels = range(1))
	
	for encoder_method in encoders:
		classifiers = [
			sclas.classifier.ML(
				mlparams, 
				sclas.classifier.fannwrapper.FANNParams(name = "3x5", hidden_nodes = [5,5,5],
        			max_iterations = 2000, classifier="None")
			),

			]

		for classifier_method in classifiers:
			ename = encoder_method.get_name_wparams("n_components")
			cname = classifier_method.get_name()
			if cname == "SupportVectorClassifier": cname = "%s-gamma=%s" % (cname, classifier_method.params['gamma'])
			if cname == "ML": cname = "%s-config=%s%s" % (cname, classifier_method.mlparams.name, classifier_method.toolparams.name)
			methodname = "%s+%s" % (ename, cname)
			scl = sclas.SClas(methodname)
			
#===================================================================================================
# Training
#===================================================================================================
			scl.train_encoder(encoder_method, traindata, skipdone=True)
			scl.train_classifier(classifier_method, traindata, truth_train, n=8, ncpu=8, skipdone=True)
			
#===================================================================================================
# Testing the versability of the training
#===================================================================================================
			nclass = np.size(np.unique(truth_train))
			for tort, data, truth in zip(["training", "test"], [traindata, testdata],[truth_train, truth_test]):
				print '**** %s *****' % (tort)
				
				catalogs, _ = scl.predict(data)
				
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
				
				print 'w.  avg\tFailures:', np.size(np.nonzero(wavg - truth))/float(len(truth)) * 100, \
					'%\t F1 = ', f1wavg.sum() / nclass
				print 'Median\tFailures:', np.size(np.nonzero(medi - truth))/float(len(truth)) * 100, \
					'%\t F1 = ', f1medi.sum() / nclass
				print 'mean\tFailures:', np.size(np.nonzero(mea - truth))/float(len(truth)) * 100, \
					'%\t F1 = ', f1mea.sum() / nclass
				
				outcat = medi
			
			sclas.utils.writepickle(f1medi, os.path.join(scl.workdir, "data", "f1score_lambda%d.pkl" % Lambda))
			
			sclas.utils.mkdir(outfoler)
			output = open(os.path.join(outfoler, outfile),"a") 
			print >> output, methodname, '\t', "wavg",'\t', f1wavg.sum() / nclass
			print >> output, methodname, '\t', "Medi",'\t', f1medi.sum() / nclass
			print >> output, methodname, '\t', "Mean",'\t', f1mea.sum() / nclass
			output.close()
			
			del scl