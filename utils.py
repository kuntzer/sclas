"""
Here are a few utils
"""

import os
import cPickle as pickle
import astropy.io.fits
import gzip
import numpy as np
import matplotlib.colors as colors

import logging
logger = logging.getLogger(__name__)

def writepickle(obj, filepath, protocol = -1):
	"""
	I write your python object obj into a pickle file at filepath.
	If filepath ends with .gz, I'll use gzip to compress the pickle.
	Leave protocol = -1 : I'll use the latest binary protocol of pickle.
	"""
	if os.path.splitext(filepath)[1] == ".gz":
		pkl_file = gzip.open(filepath, 'wb')
	else:
		pkl_file = open(filepath, 'wb')
	
	pickle.dump(obj, pkl_file, protocol)
	pkl_file.close()
	logger.info("Wrote %s" % filepath)
	
def readpickle(filepath):
	"""
	I read a pickle file and return whatever object it contains.
	If the filepath ends with .gz, I'll unzip the pickle file.
	"""
	if os.path.splitext(filepath)[1] == ".gz":
		pkl_file = gzip.open(filepath,'rb')
	else:
		pkl_file = open(filepath, 'rb')
	obj = pickle.load(pkl_file)
	pkl_file.close()
	logger.info("Read %s" % filepath)
	return obj


def fromfits(filepath):
	"""
	Read simple 1-hdu FITS files -> numpy arrays, so that the indexes [x,y] follow the orientations of
	x, y on ds9, respectively.
	"""
	a = astropy.io.fits.getdata(filepath).transpose()
	logger.info("Read FITS images %s from file %s" % (a.shape, filepath))
	return a
	

def tofits(a, filepath):
	"""
	Writes a simply 2D numpy array to FITS, same convention.
	"""
	
	if os.path.exists(filepath):
		logger.warning("File %s exists, I will overwrite it!" % (filepath))

	astropy.io.fits.writeto(filepath, a.transpose(), clobber=1)
	logger.info("Wrote %s array to %s" % (a.shape, filepath))


def mkdir(somedir):
	"""
	A wrapper around os.makedirs.
	:param somedir: a name or path to a directory which I should make.
	"""
	if not os.path.isdir(somedir):
		os.makedirs(somedir)
		
def rmsd(x, y):
	"""
	Returns the RMSD between two numpy arrays
	(only beginners tend to inexactly call this the RMS... ;-)
	
	http://en.wikipedia.org/wiki/Root-mean-square_deviation
	
	This function also works as expected on masked arrays.	
	"""
	return np.sqrt(np.mean((x - y)**2.0))

def rmsd_delta(delta):
	"""
	Idem, but directly takes the estimation errors.
	Useful, e.g., to pass as reduce_C_function for hexbin plots.
	"""
	return np.sqrt(np.mean(np.asarray(delta)**2.0))

def around(a, nearest=0.5, amin=None, amax=None):
	"""
	:param a: numpy array
	:param nearest: nearest value to round to
	:param amin: minimum value allowed, if `None`, there is no bottom value
	:param amax: maximum value allowed, if `None`, there is no top value
	
	Rounds an array to `nearest` value. Ex: 
	>>> a = np.random.random(size=10)
	>>> around(a, 0.5)
	array([ 0. ,  0.5,  1. ,  0.5,  0.5,  0.5,  0. ,  0. ,  0.5,  0.5])
	"""
	res = np.round(a/nearest)*nearest
	if amin is None and amax is None:
		return res
	else:
		return np.clip(res, amin, amax)
	
def anearest(a, reference, distance_threshold=None, error_symbol=99):
	"""
	Returns the nearest entry in the vector `reference` for the elements contained in vector `a`.
	A maximal distance can be specified. If an element in the input array `a` is too far away from
	the closest reference element, it will be take the `error_symbol` value in the output vector.
	
	:param a: input vector data
	:param reference: reference vector data, the elements of the output array are taken from this
	:param distance_threshold: optional, the maximal distance allowed. If the distance is larger, the
		`error_symbol` value will be used. If `distance_thershold` is set to `None`, there is no
		maximum distance.
	:param error_symbol: optional, the value to be taken in case of a too large distance with the
		nearest value in `reference`. If `distance_threshold` is `None`, this parameter is useless.
	"""
	# This is _not_ the best implementation possible. Possibly something along the lines of:
	# ( np.tile(a, np.size(reference) - np.tile(reference, np.size(a) ).argmin(axis=1)
	# should be faster, but would use more memory. At the end of the day, this function should not
	# be called often, so 

	out = np.empty_like(a)
	for ii, value in enumerate(a):
		idv = find_nearest(reference, value)
		out[ii] = reference[idv]
		
	if not distance_threshold is None:
		out[np.abs(out-a) > distance_threshold] = error_symbol

	return out
	#return find(reference, a)
	
def normalise(arr):
	"""
	Normalise between 0 and 1 the data, see `func:unnormalise` for the inverse function
	
	:param arr: the data to normalise
	
	:returns: normed array, lowest value, highest value
	"""
	low = np.amin(arr)
	high = np.amax(arr)
	normed_data = (arr - low) / (high - low)
	return normed_data, low, high

def unnormalise(arr, low, high):
	return arr * (high - low) + low

def find_nearest(array, value):
	''' Find nearest value in an array '''
	idx = (np.abs(array-value)).argmin()
	return idx

def mad(a, axis=None):
	"""
	Compute *Median Absolute Deviation* of an array along given axis.
	"""

	med = np.median(a, axis=axis)				# Median along given axis
	if axis is None:
		umed = med							  # med is a scalar
	else:
		umed = np.expand_dims(med, axis)		 # Bring back the vanished axis
	mad = np.median(np.absolute(a - umed), axis=axis)  # MAD along given axis

	return mad

def removekey(d, key):
	r = dict(d)
	del r[key]
	return r



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
	new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
	return new_cmap
