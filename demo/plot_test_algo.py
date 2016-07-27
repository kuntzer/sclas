import pylab as plt
import csv
import numpy as np
import sclas.plots.figures as figures
import sclas
import os

infolder = 'data'
outfolder = 'figures'
fname = "pca+fann_100classifiers_every1_54random.dat.csv"
fname = 'hst_814_step_pca+fann_100classifiers_every1.dat.csv'
fname = 'pca+fann_100classifiers_every1_lambda10_54best.dat.csv'
fname = 'hst_606_step_pca+fann_100classifiers_every1.dat.csv'
#fname = 'pca+fann_100classifiers_every2_1level.dat.csv'
note_to_fname = "median"

configs = [1]
columns_to_remove = [0,1,2,12]
columns_to_remove = [0,1,2,20]
columns_to_remove = [0,1,2,19]
ncomponents = [0]
lines_to_remove = [0]


figures.set_fancy()

data = []
with open(os.path.join(infolder, fname), 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in reader:
		data += [row]

data = np.asarray(data)
configs = data[:,configs];

configs = [c[0].split('x')[-1] for ii, c in enumerate(configs) if ii > 0]
configs = np.asarray(configs, np.float)

data = np.delete(data, columns_to_remove, 1)
xs = data[ncomponents]
print xs
xs = np.asarray(xs[0], np.float)
data = np.delete(data, lines_to_remove, 0)

data = np.asarray(data, np.float)

X, Y = np.meshgrid(xs, configs)

vmax = np.ceil(np.amax(data)*200.)/200.
vmin = np.floor(np.amin(data)*200.)/200.
vmax = np.ceil(np.amax(data)*20.)/20.
vmin = np.floor(np.amin(data)*20.)/20.
print 'f1 max:', np.amax(data), vmax
print 'f1 min:', np.min(data), vmin
print 'f1 mean:', np.mean(data)
#vmax=0.9
#vmin=0.75
"""
f=plt.figure()
ax=plt.subplot(111)
#pc=plt.contourf(X,Y,(data),[0.,0.7,0.8,.84,0.85,0.86,0.87,0.88,0.89,0.9], vmin=0.7, vmax=np.amax(data), cmap=plt.cm.Spectral)
clevels = np.linspace(0.,0.2,100)
pc=plt.contourf(X,Y,(data), 20)#, clevels, vmin=0., vmac=0.02, cmap=plt.cm.Spectral_r)
#, vmin=0., vmax=0.2, 
cbar = plt.colorbar()
#cbar.set_label(r"$\mathrm{Error\ on}\ F_1$")
#print [r for r in cbar.ax.get_yticks()]
#cbar.ax.set_yticklabels([r"$%0.03f$" % l for l in cbar.ax.get_yticks()])
plt.xlabel(r"$\mathrm{Number\ of\ components}$")
plt.ylabel(r"$\mathrm{Number\ of\ hidden\ nodes}$")
#print [r"$%0.03f$" % l for l in cbar.ax.get_yticks()]
"""
"""
f=plt.figure()
ax=plt.subplot(111)

vmax=0.032
vmin=0.025

pc=ax.matshow((data[::-1,:]), vmin=vmin, vmax=vmax, extent=[xs[0], xs[-1]+1, configs[0], configs[-1]+1])
cbar = plt.colorbar(pc)
cbar.set_label(r"$\mathrm{Error\ on}\ F_1$")
plt.xlabel(r"$\mathrm{Number\ of\ components}$")
plt.ylabel(r"$\mathrm{Number\ of\ hidden\ nodes}$")

xminorlocations = xs[::2]
ax.set_xticks(xminorlocations+0.5)
ax.set_xticklabels([r"$%d$" % l for l in xminorlocations])
ax.xaxis.tick_bottom()

yminorlocations = configs[::2]
ax.set_yticks(yminorlocations+0.5)
ax.set_yticklabels([r"$%d$" % l for l in yminorlocations])

"""
f=plt.figure()
ax=plt.subplot(111)

vmax = 0.66
vmin = 0.6

pc=ax.matshow((data[::-1,:]), vmin=vmin, vmax=vmax, cmap=plt.cm.Spectral, extent=[xs[0], xs[-1]+1, configs[0], configs[-1]+1])
cbar = plt.colorbar(pc)
cbar.set_label(r"$F_1\mathrm{\ score}$")
plt.xlabel(r"$\mathrm{Number\ of\ components}$")
plt.ylabel(r"$\mathrm{Number\ of\ hidden\ nodes}$")

xminorlocations = xs[::2]
ax.set_xticks(xminorlocations+0.5)
ax.set_xticklabels([r"$%d$" % l for l in xminorlocations])
ax.xaxis.tick_bottom()

yminorlocations = configs[::2]
ax.set_yticks(yminorlocations+0.5)
ax.set_yticklabels([r"$%d$" % l for l in yminorlocations])


sclas.utils.mkdir(outfolder)
fnamefig = fname.split('.')[0]
figures.savefig(os.path.join(outfolder, fnamefig+note_to_fname), f, True)


plt.show()