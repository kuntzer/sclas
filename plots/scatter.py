import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D

import logging
logger = logging.getLogger(__name__)

def scatter(ax, x, y, c=None, x_name=None, y_name=None, c_name=None, cmap="jet", title=None, 
		text=None, show_id_line=False, idlinekwargs=None, sidehists=False, sidehistkwargs=None, 
		errorbarkwargs=None, colorbarticks=None, colorbarticklabels=None, **kwargs):
	"""
	A simple scatter plot of data in x and y. A third array, c, gives an optional colorbar.
	
	.. note:: If you specify ``c``, this function uses matplotlib's ``scatter()``. Otherwise, the function uses ``plot()``, as
		plot is much faster for large numbers of points! The possible ``**kwargs`` change accordingly!

	:param ax: a matplotlib.axes.Axes object
	:param x: data for the x axis
	:param y: data for the y axis
	:param c: data to use for the colorbar, decides if plot() or scatter() is used.
	:param x_name: label of x axis
	:param y_name: label of y axis
	:param cmap: the color bar to use. For a scatter plot one usually wants to see every point, avoid white!
	:param title: the title to place on top of the axis.
		The reason why we do not leave this to the user is that the placement changes when sidehists is True.
	:param text: some text to be written in the figure (top left corner)
		As we frequently want to do this, here is a simple way to do it.
		For more complicated things, add the text yourself to the axes.
	:param show_id_line: draws an "identity" diagonal line
	:param idlinekwargs: a dict of kwargs that will be passed to plot() to draw the idline
	:param sidehists: adds projection histograms on the top and the left (not nicely compatible with the colorbar)
		The range of these hists are limited by your features limits. Bins outside your limits are not computed!
	:param sidehistkwargs: a dict of keyword arguments to be passed to these histograms.
		Add range=None to these if you want all bins to be computed.
	:param errorbarkwargs: a dict of keywords to be passed to errorbar()
	
	Any further kwargs are either passed to ``plot()`` (if no featc is given) or to ``scatter()``.
	
	Some commonly used kwargs for plot() are:
	
	* **marker**: default is ".", you can switch to single pixel (",") or anything else...
	* **ms**: marker size in points
	* **color**: e.g. "red"
	* **label**: for the legend

	Some commonly used kwargs for scatter() are:
	
	* **s**: marker size
	* **label**: for the legend

	By default plots will be rasterized if the catalog has more than 5000 entries. To overwrite,
	just pass rasterized = True or False as kwarg.

	"""
	
	# Some initial settings:
	if sidehistkwargs is None:
		sidehistkwargs = {}
	if errorbarkwargs is None:
		errorbarkwargs = {}
	
	if len(x) > 5000: # We rasterize plot() and scatter(), to avoid millions of vector points.
		logger.info("Plot will be rasterized, use kwarg rasterized=False if you want to avoid this")
		rasterized = True
	else:
		rasterized = False
		
	# And now, two options:
	if c is not None: # We will use scatter(), to have a colorbar

		# We prepare to use scatter, with a colorbar
		cmap = matplotlib.cm.get_cmap(cmap)
		mykwargs = {"marker":"o", "lw":0, "s":30, "cmap":cmap, "vmin":np.amin(c),
				"vmax":np.amax(c), "rasterized":rasterized, "edgecolor":"None"}
		
		# We overwrite these mykwargs with any user-specified kwargs:
		mykwargs.update(kwargs)
		
		# And make the plot:
		stuff = ax.scatter(x, y, c=c, **mykwargs)
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", "5%", pad="3%")
		cax = plt.colorbar(stuff, cax, ticks=colorbarticks)
		if not colorbarticklabels is None:
			cax.set_ticklabels(colorbarticklabels)
		if not c_name is None:
			cax.set_label(c_name)
			
	else: # We will use plot()
	
		logger.info("Preparing plain plot of %i points without colorbar" % (len(x)))
		mykwargs = {"marker":".", "ms":5, "color":"black", "ls":"None", "alpha":0.3, "rasterized":rasterized}
	
		# We overwrite these mykwargs with any user-specified kwargs:
		mykwargs.update(kwargs)
		
		# And we also prepare any errorbarkwargs
		myerrorbarkwargs = {"capthick":0, "zorder":-100, "rasterized":rasterized} 
		# Different from the defaults for scatter() !
		myerrorbarkwargs.update(errorbarkwargs)
		
		# And now the actual plot:
		#	# Plain plot:
		ax.plot(x, y, **mykwargs)
		
	
	# We want minor ticks:
	ax.xaxis.set_minor_locator(AutoMinorLocator(5))
	ax.yaxis.set_minor_locator(AutoMinorLocator(5))
	
	if sidehists:
		
		# By default, we want to limit the "binning" of the actual histograms (not just their display) to the specified ranges.
		# However, this needs some special treatment for the case when the "low" or "high" are set to None.
		if np.amin(x) is not None and np.amax(x) is not None: 
			histxrange = (np.amin(x), np.amax(x))
		else:
			histxrange = None
		if np.amin(y) is not None and np.amax(y) is not None: 
			histyrange = (np.amin(y), np.amax(y))
		else:
			histyrange = None
		# If you do not like this behaviour, simply set the sidehistkwarg "range" to None !
		
		
		# Same as for kwargs: we first define some defaults, and then update these defaults:
		
		mysidehistxkwargs = {"histtype":"stepfilled", "bins":100, "ec":"none", "color":"gray", "range":histxrange}
		mysidehistxkwargs.update(sidehistkwargs)
		mysidehistykwargs = {"histtype":"stepfilled", "bins":100, "ec":"none", "color":"gray", "range":histyrange}
		mysidehistykwargs.update(sidehistkwargs)
		
		# We prepare the axes for the hists:
		divider = make_axes_locatable(ax)
		axhistx = divider.append_axes("top", 1.0, pad=0.1, sharex=ax)
		axhisty = divider.append_axes("right", 1.0, pad=0.1, sharey=ax)
		
		# And draw the histograms		
		axhistx.hist(x, **mysidehistxkwargs)
		axhisty.hist(y, orientation='horizontal', **mysidehistykwargs)
		
		# Hiding the ticklabels
		for tl in axhistx.get_xticklabels():
			tl.set_visible(False)
		for tl in axhisty.get_yticklabels():
			tl.set_visible(False)
		
		# Reducing the number of major ticks...
		#axhistx.yaxis.set_major_locator(MaxNLocator(nbins = 2))
		#axhisty.xaxis.set_major_locator(MaxNLocator(nbins = 2))
		# Or hide them completely
		axhistx.yaxis.set_ticks([]) # or set_ticklabels([])
		axhisty.xaxis.set_ticks([])
	
		if title:
			axhistx.set_title(title)
		
	else:
		if title:
			ax.set_title(title)
	
	if show_id_line: # Show the "diagonal" identity line
	
		# It would be nice to get this working with less code
		# (usign get_lims and axes transforms, for instance)
		# But in the meantime this seems to work fine.
		# It has to be so complicated to keep the automatic ranges working if low and high are None !
		
		# For "low":
		if np.amin(x) is None or np.amin(y) is None: # We use the ...
			minid = max(np.min(x), np.min(y))
		else:
			minid = max(np.amin(x), np.amin(y))
		# Same for "high":
		if np.amax(x) is None or np.amax(y) is None: # We use the ...
			maxid = min(np.max(x), np.max(y))
		else:
			maxid = min(np.amax(x), np.amax(y))
			
		if idlinekwargs == None:
			idlinekwargs = {}
		myidlinekwargs = {"ls":"--", "color":"gray", "lw":1}
		myidlinekwargs.update(idlinekwargs)	
		
		# And we plot the line:
		ax.plot((minid, maxid), (minid, maxid), **myidlinekwargs)


	ax.set_xlim(np.amin(x), np.amax(x))
	ax.set_ylim(np.amin(y), np.amax(y))
	if not x_name is None:
		ax.set_xlabel(x_name)
	if not y_name is None:
		ax.set_ylabel(y_name)
	"""
	ax.set_xlim([-4,3])
	ax.set_ylim([-5, 3])
		
	ax.yaxis.set_ticklabels([])
	ax.set_ylabel('')
	"""	
	ax.grid(True)


	# Finally, we write the text:
	if text:
		ax.annotate(text, xy=(0.0, 1.0), xycoords='axes fraction', xytext=(8, -8), textcoords='offset points', ha='left', va='top')
