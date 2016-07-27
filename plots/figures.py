"""
This module provides several specific functions to save figures nicely. `fancy` means latex 
interpreter and generates *.eps and *.pdf
"""

import numpy as np
import matplotlib

def savefig(fname,fig,fancy=False,pdf_transparence=False):
	import os
	import subprocess

	directory=os.path.dirname(os.path.abspath(fname))
	if not os.path.exists(directory):
		os.makedirs(directory)

	fig.savefig(fname+'.png',dpi=300)

	if fancy: 
		fig.savefig(fname+'.pdf',transparent=pdf_transparence)
		#fig.savefig(fname+'.eps',transparent=True)
		#os.system("epstopdf "+fname+".eps")
		command = 'pdfcrop %s.pdf' % fname
		subprocess.check_output(command, shell=True)
		os.system('mv '+fname+'-crop.pdf '+fname+'.pdf')
	

def set_fancy():
	from matplotlib import rc
	#rc('font',**{'family':'serif','serif':['Palatino'],'size':16})
	rc('font',**{'family':'serif','size':16})
	rc('text', usetex=True)



def cmap_map(function,cmap):
	""" Applies function (which should operate on vectors of shape 3:
	[r, g, b], on colormap cmap. This routine will break any discontinuous	 points in a colormap.
	"""
	cdict = cmap._segmentdata
	step_dict = {}
	# Firt get the list of points where the segments start or end
	for key in ('red','green','blue'):		 step_dict[key] = map(lambda x: x[0], cdict[key])
	step_list = sum(step_dict.values(), [])
	step_list = np.array(list(set(step_list)))
	# Then compute the LUT, and apply the function to the LUT
	reduced_cmap = lambda step : np.array(cmap(step)[0:3])
	old_LUT = np.array(map( reduced_cmap, step_list))
	new_LUT = np.array(map( function, old_LUT))
	# Now try to make a minimal segment definition of the new LUT
	cdict = {}
	for i,key in enumerate(('red','green','blue')):
		this_cdict = {}
		for j,step in enumerate(step_list):
			if step in step_dict[key]:
				this_cdict[step] = new_LUT[j,i]
			elif new_LUT[j,i]!=old_LUT[j,i]:
				this_cdict[step] = new_LUT[j,i]
		colorvector=  map(lambda x: x + (x[1], ), this_cdict.items())
		colorvector.sort()
		cdict[key] = colorvector

	return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

