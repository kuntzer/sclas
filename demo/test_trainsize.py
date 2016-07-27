"""
This demo must be run after `run.py`. It shows how to use the trainsize function.
"""

import sclas
import os
import numpy as np
import pylab as plt

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)


demo = sclas.SClas('demo')

#===================================================================================================
# Load the data
#===================================================================================================
datapklfname = os.path.join("../../data", "train_data.pkl")
traindata, noise_maps = sclas.utils.readpickle(datapklfname)
datapklfname = os.path.join("../../data", "test_data.pkl")
testdata, noise_maps = sclas.utils.readpickle(datapklfname)

truth_train = np.loadtxt("../../data/euclid_noiseless/star_truth.dat")[:1000,1]
truth_test = np.loadtxt("../../data/euclid_noiseless/star_truth.dat")[1000:3000,1]

#===================================================================================================
# Run the trainsize function
#===================================================================================================
# Change the name of the directory to avoid destroying old stuff:
demo.classifier.set_workdir(os.path.join(demo.classifier.workdir,"test_trainsize"))

traindata = demo.encoder.encode(traindata) 
testdata = demo.encoder.encode(testdata) 

sclas.diagnostics.test_trainsize(demo.classifier, 'f1score', traindata, truth_train, testdata, 
								truth_test, ncpu=1)
plt.show()
