import numpy as np


def plot(ax, x, y=None, x_name=None, y_name=None, title=None, text=None, 
		labels=None, legendkwargs=None, colors=None, **kwargs):
	"""
	A simple plot of data in x and y. A third array

	:param ax: a matplotlib.axes.Axes object
	:param x: a list of data for the x axis
	:param y: a list of data for the y axis
	:param title: the title to place on top of the axis.
	:param text: some text to be written in the figure (top left corner)
		As we frequently want to do this, here is a simple way to do it.
		For more complicated things, add the text yourself to the axes.
	:param labels: a list of labels
	:param colors: a list of colors
	:param legendkwargs: kwargs for the legend
	
	Any further kwargs are either passed to ``plot()`` (if no featc is given) or to ``scatter()``.
	
	Some commonly used kwargs for plot() are:
	
	* **marker**: marker type
	* **ms**: marker size in points
	* **ls**: linestyle
	* **lw**: linewdith
	* **color**: e.g. "red"
	* **label**: for the legend
	
	.. warning:: Add errorbars capacity! 
	
	"""
	
	nd = np.shape(x)[0]
	if not y is None and not nd == np.shape(y)[0]:
		raise IndexError("Number of plots are not the same in the x and y directions")
	if not labels is None and not nd == np.shape(labels)[0]:
		raise IndexError("Number of labels does not correspond to number of plots")
	if not colors is None and not nd == np.shape(colors)[0]:
		raise IndexError("Number of colors does not correspond to number of plots")
	#TODO: Check labels too
				
	mykwargs = {"ls":"-", "alpha":0.8, "lw":2}
	# We overwrite these mykwargs with any user-specified kwargs:
	mykwargs.update(kwargs)
	
	if nd == 1:
		if y is None:
			ax.plot(x[0], **mykwargs)
		else:
			ax.plot(x[0], y[0], **mykwargs)

	else:
		if labels is None:
			labels = [None] * nd
		if colors is None:
			colors = [None] * nd
		
		for ii, lx in enumerate(x):
			if y is None:
				ax.plot(lx, label=labels[ii], c=colors[ii], **mykwargs)
			else:
				ax.plot(lx, y[ii], label=labels[ii], c=colors[ii], **mykwargs)
				
	if not title is None:
		ax.set_title(title)

	if not x_name is None:
		ax.set_xlabel(x_name)
	if not y_name is None:
		ax.set_ylabel(y_name)

	ax.grid(True)
	
	if not labels is None:
		mykwargs = {"loc":"best"}
		# We overwrite these mykwargs with any user-specified kwargs:
		if not legendkwargs is None:
			mykwargs.update(legendkwargs)
		ax.legend(**mykwargs)
		

	# Finally, we write the text:
	if text:
		ax.annotate(text, xy=(0.0, 1.0), xycoords='axes fraction', xytext=(8, -8), textcoords='offset points', ha='left', va='top')
