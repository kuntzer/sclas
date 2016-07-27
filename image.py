"""
A suite of algorithms to handle images
"""

import numpy as np
from scipy import ndimage

def block_mean(ar, fact):
	"""
	Rebins an array to a smaller size by making a block mean of the input image
	
	:param ar: the image
	:type ar: numpy array
	:param fact: the reduction factor
	
	:returns: an numpy array
	"""
	
	assert isinstance(fact, int), type(fact)
	sx, sy = ar.shape
	X, Y = np.ogrid[0:sx, 0:sy]
	regions = sy/fact * (X/fact) + Y/fact
	res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
	res.shape = (sx/fact, sy/fact)
	return res

def downsample(a, newshape):
	"""
	Rebins ndarray data into a smaller ndarray of the same rank whose dimensions
	are factors of the original dimensions. eg. An array with 6 columns and 4 rows
	can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
	example usages:
	
	>>> a=rand(6,4); b=downsample(a,(3,2))
	>>> a=rand(6); b=downsample(a,2)
	"""
	shape = a.shape
	lenShape = len(shape)
	factor = np.asarray(shape)/np.asarray(newshape)
	evList = ['a.reshape('] + \
		 ['newshape[%d],factor[%d],'%(i,i) for i in xrange(lenShape)] + \
		 [')'] + ['.sum(%d)'%(i+1) for i in xrange(lenShape)]
	return eval(''.join(evList))
