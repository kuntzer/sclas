import numpy as np
import pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

def matshow(ax, data, x_name=None, y_name=None, x_n=None, y_n=None, 
		x_tickslabels=None, y_tickslabels=None, inversed=False, cmap=plt.cm.gray_r, 
		colorbar=False, colorbarticks=None, colorbarticklabels=None, c_name=None, **kwargs):
	"""
	A simple matrix showing plot, based on 
	http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.matshow

	:param ax: a matplotlib.axes.Axes object
	:param data: the numpy array matrix to plot
	:param x_name: label of x axis
	:param y_name: label of y axis
	:param x_n: number of elements in x direction, by default take shape of ``data``
	:param y_n: number of elements in y direction, by default take shape of ``data``
	:param cmap: the color bar to use. Default: gray_r
	:param title: the title to place on top of the axis.
	:param text: some text to be written in the figure (top left corner)
		As we frequently want to do this, here is a simple way to do it.
		For more complicated things, add the text yourself to the axes.
	:param x_tickslabels: x axis tick labels
	:param y_tickslabels: y axis tick labels 
	:param inversed: if True inverses the y-axis to get the classical representation of a matrix
	:param colorbar: If True includes a colorbar
	:param c: data to use for the colorbar
	
	Any further kwargs are either passed to ``matshow()``.
	
	"""
	
	stuff = ax.matshow(data, cmap=cmap, origin='lower', **kwargs)
	if x_name is not None:
		ax.set_xlabel(x_name)
	if y_name is not None:
		ax.set_ylabel(y_name)
		#ax.set_ylabel("")
		
	if x_n is None:
		x_n = np.shape(data)[1]
	if y_n is None:
		y_n = np.shape(data)[0]
		
	ax.set_xticks(np.arange(x_n))
	ax.set_yticks(np.arange(y_n))
	
	if x_tickslabels is not None:
		ax.set_xticklabels(x_tickslabels)
	if y_tickslabels is not None:
		ax.set_yticklabels(y_tickslabels)
		#ax.set_yticklabels([''] * len(y_tickslabels))

	if inversed:
		ax.invert_yaxis()
	else:
		ax.xaxis.tick_bottom()
		
	if colorbar:

		# And make the plot:
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", "5%", pad="3%")
		cax = plt.colorbar(stuff, cax, ticks=colorbarticks)
		if not colorbarticklabels is None:
			cax.set_ticklabels(colorbarticklabels)
		if not c_name is None:
			cax.set_label(c_name)
		
	
