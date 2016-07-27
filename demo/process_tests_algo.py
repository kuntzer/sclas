import numpy as np
import sclas
import csv

data = []

encoders = ["PCA", "ICA", "NMF", "LLE"]
encoders = ["PCA"]
ncomponents = range(5,21)

ncl = ncomponents #+ [100]
classifiers = ["ML-config=FANN6063x%d" % nc for nc in range(5,31)]

fname = 'data/pca+fann_100classifiers_every1_54best.dat'
fname = 'data/hst_606_step_pca+fann_100classifiers_every1.dat'
#fname = 'data/pca+fann_100classifiers_every2_1level.dat'

tab = []
res = np.zeros([len(encoders)*3, len(classifiers)*len(ncomponents)])

with open(fname, 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
	for row in reader:
		data += [row]

count = 0
for encoder in encoders:
	for ncomponent in ncomponents:
		for classifier in classifiers:
			for row in data:
				if encoder in row[0] and "n_components=%d+" % ncomponent in row[0] and classifier in row[0]:
					count += 1
					encoder = row[0].split("-")[0]
					
					classifier = row[0].split("+")[1]#.split("-")[0]
					kind = row[1][:-1]
					f1 = row[2]
					#print encoder, row[0], "n_components=%d" % ncomponent, row[0], classifier, row[0]
					print count, encoder,ncomponent, classifier, kind, f1
					#print encoder, ncomponent, classifier, clas_params, kind, f1
					tab.append([encoder, ncomponent, classifier[:-1], kind, f1])
							
print 
print
tab = np.asarray(tab)
for t in tab:
	print t
print 

output=open("%s.csv" % fname,"w") 

""" Many encoders, few classifiers
print >> output,  'f1 kind', 
for c in classifiers:
	for ii, nc in enumerate(ncomponents):
		if ii==0: print >> output,  c,
		else: print >> output, '-',

print >> output,  ''
print >> output,  'encoder kind',
for c in classifiers:
	for nc in ncomponents:
		print >> output,  nc,
print >> output, ''
print '----------------'
for encoder in encoders:
	et = tab[tab[:,0]==encoder]
	for kind in ["errMean"]:#, "errwavg", "errMedi"]:#["wavg", "Medi", "Mean"]:
		kt = et[kind == et[:,3]]
		print >> output,  encoder, kind, 
		for classifier in classifiers:
			ct = kt[classifier == kt[:,2]]
			for ncomponent in ncomponents:
				try:
					nct = ct[ct[:,1]=='%s' % ncomponent]
					print >> output, nct[0][-1],
				except IndexError:
					print >> output, "-", kind, ncomponent
				if np.shape(nct)[0] != 1:
					print 'encoder:', encoder,'n=',ncomponent,'kind', kind
					print 'classifier', classifier
					print nct;
					raise ValueError("Too many or too few components")
		print >> output, ''
output.close()
"""

""" Few encoders, many classifiers """
print >> output,  'encoder classifier kind',
for nc in ncomponents:
	print >> output,  nc,
print >> output, ''
print '----------------'
for encoder in encoders:
	et = tab[tab[:,0]==encoder]
	for classifier in classifiers:
		ct = et[classifier == et[:,2]]
		for kind in ["Medi"]:#, "errwavg", "errMedi"]:#["wavg", "Medi", "Mean"]:
			kt = ct[kind == ct[:,3]]
			print >> output,  encoder, classifier, kind, 
			for ncomponent in ncomponents:
				try:
					nct = kt[kt[:,1]=='%s' % ncomponent]
					print >> output, nct[0][-1],
				except IndexError:
					print >> output, 0.,
				'''if np.shape(nct)[0] != 1:
					print 'encoder:', encoder,'n=',ncomponent,'kind', kind
					print 'classifier', classifier
					print nct;
					raise ValueError("Too many or too few components")'''
			print >> output, ''
output.close()
