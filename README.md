# sclas
## Classification tool based on a single wide-band filter image

This code was developed to identify spectral classes in [Euclid](http://www.euclid-ec.org/) images based on only one image. The results are described in a peer-reviewed paper _Stellar classification from single-band imaging using machine learning_ by Kuntzer, T.; Tewes, M. and Courbin, F. published in 2016.

## Latest release

The latest release can be found [here](sclas-1.0beta.tar.gz).

## Abstract of the accompagning paper
Information on the spectral types of stars is of great interest in view of the exploitation of space-based imaging surveys. In this article, we investigate the classification of stars into spectral types using only the shape of their diffraction pattern in a single broad-band image. We propose a supervised machine learning approach to this endeavour, based on principal component analysis (PCA) for dimensionality reduction, followed by artificial neural networks (ANNs) estimating the spectral type. Our analysis is performed with image simulations mimicking the Hubble Space Telescope (HST) Advanced Camera for Surveys (ACS) in the F606W and F814W bands, as well as the Euclid VIS imager. We first demonstrate this classification in a simple context, assuming perfect knowledge of the point spread function (PSF) model and the possibility of accurately generating mock training data for the machine learning. We then analyse its performance in a fully data-driven situation, in which the training would be performed with a limited subset of bright stars from a survey, and an unknown PSF with spatial variations across the detector. We use simulations of main-sequence stars with flat distributions in spectral type and in signal-to-noise ratio, and classify these stars into 13 spectral subclasses, from O5 to M5. Under these conditions, the algorithm achieves a high success rate both for Euclid and HST images, with typical errors of half a spectral class. Although more detailed simulations would be needed to assess the performance of the algorithm on a specific survey, this shows that stellar classification from single-band images is well possible. 

## Accessing / citing the paper
[DOI](http://dx.doi.org/10.1051/0004-6361/201628660)

[ADS entry](http://adsabs.harvard.edu/abs/2016A%26A...591A..54K)

## Running the code

In the latest release, there is a direction demo/ containing working examples based on non-astrophysical data.
