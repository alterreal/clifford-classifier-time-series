#############################################################################################################################
#                                                                                                                           #
#                    **Example of CliffordClassifier for multivariate time series classification**                          #
#                                                                                                                           #
# The algorithm is implemented as described in the paper "A Geometric Model-Based Approach to Hand Gesture Recognition",    #
# along with some additional features.                                                                                      #
#                                                                                                                           #
# Author: Alexandre Calado                                                                                                  #
#                                                                                                                           #
#############################################################################################################################

import cliffordTS
import numpy as np
from pyts.datasets import *
from sklearn.metrics import accuracy_score

#Fetch sample dataset from the UEA multivariate time series classification archive (https://timeseriesclassification.com) 
#using pyts (https://pyts.readthedocs.io/en/stable/). In this example let us use the "PenDigits" Dataset
#WARNING 1: Note that these algorithms do not work on univariate time series. If so, a transform, such as Taken's Embedding 
#must be used to project the data onto a higher dimensional space before further processing
#WARNING 2: Note that, if using numpy, the dataset must have the following shape: 
#(number of instances, number of dimensions, time series length)
#WARNING 3: Note that the number of 2-blades returned by the external product between two vectors is given by the binomial
#coefficent C(D, 2), where D is the dimension of the time series. So, the higher the dimension, the much higher the number 
#of resulting 2-blades will be, which makes the algorithm slower
dataset = fetch_uea_dataset('PenDigits')

#Get train and test sets
X_train = dataset['data_train']
X_test = dataset['data_test']
y_train = dataset['target_train']
y_test = dataset['target_test']

#Choose the dimensionless distance metric (uC1 or uC2)
metric = "uC1"

#Choose the desired number of points N after resample (ommit this parameter or set it to "None" if you do not wish to resample). 
#Note that if you do not resample, you will lose time series length invariance
N_points = None

#DO NOT forget to scale the dataset so that all values from the train set fall in the range [-1,1]. Otherwise the computed 
#point-wise metrics will not fall within the same range
tr_max = np.max(np.max(X_train, axis = 2,  keepdims = True),0, keepdims = True)
tr_min = np.min(np.min(X_train, axis = 2,  keepdims = True),0,  keepdims = True)

X_train_scaled = 2*(X_train-tr_min)/(tr_max-tr_min)-1
X_test_scaled = 2*(X_test-tr_min)/(tr_max-tr_min)-1


clf = cliffordTS.CliffordClassifier(dist = metric, resample_length = N_points) #Build the classifier
clf.fit(X_train_scaled,y_train) #Train the classifier (preprocess train set)
y_hyp,_ = clf.predict(X_test_scaled) #Classify test data


accuracy = accuracy_score(y_test, y_hyp) #Compute accuracy
print("Accuracy obtained with " + metric + ": " + str(accuracy*100) + "%")