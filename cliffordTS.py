#########################################################################################################################
#                                                    +++ READ ME +++                                                    #
#                                                                                                                       #
# This code regards the Python implementation of uC1 and uC2 geometric algebra-based models (originally implemented in  #
# matlab) described in:                                                                                                 #
#                  ----------------------------------------------------------------------------------                   #
#                  |A. Calado, P. Roselli, V. Errico, N. Magrofuoco, J. Vanderdonckt, and G. Saggio,|                   #
#                  |“A Geometric Model-Based Approach to Hand Gesture Recognition”,                 |                   #
#                  |IEEE Trans. Syst. Man, Cybern. Syst., vol. 52, no. 10, pp. 6151–6161, 2022.     |                   #
#                  ----------------------------------------------------------------------------------                   #
# If you use uC1 and uC2 in you work, please CITE this paper.                                                           #
# Moreover, if there is any detail that is not understood, this paper should be the first resource to be checked.       #
#                                                                                                                       #
# Although these models were used for sign language recognition in the paper above, they can be used for classifying    #
# time series of any dimension (except one-dimensional, as the algorithms rely on the shape of triangles formed by      #
# consecutive vectors). Thus, we want to make our algorithms publicly available to any researcher working on time series# 
# classification, as it can be useful for specific problems. Contributions for making the code more efficient are also  #
# welcome.                                                                                                              #
#                                                                                                                       #
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                                                                                                                       #
# HOW TO USE:                                                                                                           #
#                                                                                                                       #
# The classifier object is initialized with:                                                                            #
# --------------------------------------------------------------------------                                            #
# |CliffordClassifier(dist, resample_length, shape_computation, w0, w1, w2)|                                            #
# --------------------------------------------------------------------------                                            #
# -> dist (string):                         "uC1" or "uC2", depending on the multivector metric you want to use         #
#                                                                                                                       #
# -> resample_length (int, OPTIONAL):       nº of points after performing Uniformly Linearly Interpolated Resampling    #
#                                           If set to "None", no resampling is applied. Must be higher than zero.       #
#                                           It is set to "None" by default.                                             #
#                                                                                                                       #
# -> shape_computation (string, OPTIONAL):  "3_consec_points" or "centroid". "3_consec_points" computes the bivectors   # 
#                                           formed by each three consecutive points along the time series, as done in   #
#                                           the paper referenced above. "centroid" computes the bivectors formed by the #
#                                           time series centroid and each two consecutive points along the time series  #
#                                           It is set to "3_consec_points" by default.                                  #
#                                                                                                                       #
# -> w0 (float, OPTIONAL)                   Can be varied from 0 to 1. Weight quantifying the contribution of pwD to    #
#                                           the uC1 (or uC2) metric. Set to 1 by default.                               #
#                                                                                                                       #
# -> w1 (float, OPTIONAL)                   Can be varied from 0 to 1. Weight quantifying the contribution of pwVD (or  #
#                                           pwuED) to the uC1 (or uC2) metric. Set to 1 by default.                     #
#                                                                                                                       #
# -> w2 (float, OPTIONAL)                   Can be varied from 0 to 1. Weight quantifying the contribution of pwu2ED    #
#                                           to the uC metric. Set to 1 by default.                                      #
#                                                                                                                       #
#                                                                                                                       #
# The classifer can be "trained" with the method:                                                                       #
# -----------------------                                                                                               #
# |fit(X_train, y_train)|                                                                                               #
# -----------------------                                                                                               #
# -> X_train (numpy array of floats):       Train set data (scaled between -1 and 1) stored in a 3D matrix of shape:    #
#                                           (#instances, #dimensions, times series length)                              #
#                                                                                                                       #
# -> y_train (numpy array of ints):         Train labels stored in a vector                                             #
#                                                                                                                       #
#                                                                                                                       #
#                                                                                                                       #
# The test set can be classified with the method:                                                                       #
# -------------------------                                                                                             #
# |predict(X_test, y_test)|                                                                                             #
# -------------------------                                                                                             #
# -> X_test (numpy array of floats):       Test set data (scaled between -1 and 1, use train set min and max)           #
#                                           stored in a 3D matrix of shape:                                             #
#                                           (#instances, #dimensions, times series length)                              #
#                                                                                                                       #
# -> y_test (numpy array of ints):         Train labels stored in a vector                                              #
#                                                                                                                       #
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                                                                                                                       #
# AUTHORS: Alexandre Calado and Paolo Roselli                                                                           #
#                                                                                                                       #
#########################################################################################################################

import numpy as np
from numpy.linalg import norm
from numpy.linalg import det
from scipy import special 

#Classifier Object
class CliffordClassifier:  
    
    def __init__(self, dist, resample_length = None, shape_computation = '3_consec_points', w0 = 1, w1 = 1, w2 = 1):
        #initialize class objects
        
        #Distance metric to be used
        self.dist = dist
        #Method used to calculate the XS in a time series
        self.shape_computation = shape_computation
        self.resample_length = resample_length
        #weights for pwD, pwuVD and pwu2VD metrics
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2

    #Preprocesses the features by computing "shapes" and "shape of shapes"
    def __clifford_preprocess(self, dist, X, shape_computation):
        
        X_len = X.shape[0] #number of instances in X
        
        if (shape_computation == '3_consec_points'):
            XS = np.zeros((X_len, self.D_shape, self.M-2))
            XSS = np.zeros((X_len, self.D_shape2 , self.M-3))

            #Compute vectors from consecutive scalars
            Xv = X[:,:,1:] - X[:,:,:-1]
            
            den = norm(Xv[:,:,:-1],axis=1)*norm(Xv[:,:,1:],axis=1)
            den[den == 0] = 1
            
            #Compute normalized dot products between each two consecutive vectors
            XS[:,0,:] = np.sum(Xv[:,:,:-1]*Xv[:,:,1:],axis=1)/den
            
            #Compute the normalized bivectors using the external product of two consecutive vectors along the time series
            for j in range(X_len):
                for i in range(self.M-2):
                    XS[j,1:,i] = self.__compute_bivector(Xv[j,:,i],Xv[j,:,i+1])/den[j,i]    

            
            if dist == 'uC2':
                svv = XS[:,:,1:] - XS[:,:,:-1] 
                
                den = norm(svv,axis=1)
                den[den == 0] = 1
                
                XSS[:,0,:] = np.sum(XS[:,:,:-1]*svv,axis=1)/den

                for j in range(X_len):
                    for i in range(self.M-3):
                        XSS[j,1:,i] = self.__compute_bivector(XS[j,:,i],svv[j,:,i])/den[j,i] 
            
        elif (shape_computation == 'centroid'):
            
            XS = np.zeros((X_len, self.D_shape, self.M-1))
            XSS = np.zeros((X_len, self.D_shape2, self.M-2))
            
            #Compute vectors from consecutive scalars (Xv)
            Xv = X[:,:,1:] - X[:,:,:-1]
            
            #Compute vectors between each time series's centroid and points (GX)
            G = np.sum(X, axis = 2)/self.M
            GX = X - G[:,:, np.newaxis]
            GX = GX[:,:,:-1]
            
            den = norm(Xv,axis=1)*norm(GX,axis=1)
            den[den == 0] = 1
            
            #Compute normalized dot products formed by consecutive GX and Xv
            XS[:,0,:] = np.sum(Xv*GX,axis=1)/den
            #Compute the normalized bivectors formed by consecutive GX and Xv
            for j in range(X_len):
                for i in range(self.M-1):
                    XS[j,1:,i] = self.__compute_bivector(Xv[j,:,i],GX[j,:,i])/den[j,i]  

            
            if dist == 'uC2':
                svv = XS[:,:,1:] - XS[:,:,:-1] 
                den = norm(svv,axis=1)
                den[den == 0] = 1
                
                XSS[:,0,:] = np.sum(XS[:,:,:-1]*svv,axis=1)/den
                for j in range(X_len):
                    for i in range(self.M-2):
                        XSS[j,1:,i] = self.__compute_bivector(XS[j,:,i],svv[j,:,i])/den[j,i]  

        else: 
            print("Please select a valid shape computation")

        return XS, XSS


    #Computes the bivector from two n-dimensional vectors using external product (a^b)
    def __compute_bivector(self, a,b):
        D = len(a)
        A = np.vstack((a,b))
        #Compute all possible minors from matrix A with submatrices sizes of 2x2
        res = [det(np.vstack((A[:,j],A[:,j+i])).T) for i in range(1,D) for j in range(D-i)]
        return res

    #Uniformly Linearly Interpolated Resampling for D-dimensional Trajectories
    def __uniform_interp(self, x, ndp):
        rv = x[:,1:]-x[:,:-1]
        lambda_ = np.append(0,np.cumsum(np.linalg.norm(rv,axis=0)))
        lambda_N = sum(np.linalg.norm(rv,axis=0))
        h = np.multiply(np.ones((ndp-1, rv.shape[1])),np.asarray(range(1,ndp))[:,np.newaxis])
        term1 = (h*lambda_N)/(ndp-1)-lambda_[:-1]
        term1[term1<0] = 0
        rv_norm = np.linalg.norm(rv,axis=0)
        rv_norm[rv_norm == 0] = 1
        res = np.hstack((x[:,0][:,np.newaxis], np.matmul((rv/rv_norm),np.minimum(term1, np.tile(np.linalg.norm(rv,axis=0), (ndp-1,1))).T) + x[:,0][:,np.newaxis]))
        return res
        
    
    #Returns the global distance between candidates in sP and training examples in sQ 
    def __compute_GD(self,sP,sQ,PS,QS,PSS,QSS):
        
        K = sP.shape[0]
        M_S = PS.shape[2]
        M_SS = PSS.shape[2]
        
        #Matrix for storing the Global distance computed between each candidate in sP (rows) 
        #and each training example in sQ (columns). Can be used for an essemble of Clifford classifiers
        GD_matrix = np.zeros((K,self.N))
        y_hyp = []
        for i in range(K):

            GD = 0
            if self.dist == 'uC1':
                pwD = np.sum(norm(sP[i,:,:] - sQ, axis = 1), axis = 1)/self.M #point-wise distance
                pwuVD = np.sum(norm(PS[i,:,:] - QS, axis = 1), axis = 1)/(M_S) #point-wise multivector distance
                GD =  self.w0*pwD + self.w1*pwuVD
                
            elif self.dist == 'uC2':
                pwD = np.sum(norm(sP[i,:,:] - sQ, axis = 1), axis = 1)/self.M #point-wise distance
                pwuVD = np.sum(norm(PS[i,:,:] - QS, axis = 1), axis = 1)/(M_S) #point-wise multivector distance
                pwu2VD = np.sum(norm(PSS[i,:,:] - QSS, axis = 1), axis = 1)/(M_SS) #second-order point-wise multivector distance
                GD =  self.w0*pwD + self.w1*pwuVD + self.w2*pwu2VD
                
            else:
                print("Please select a valid distance metric")
                   
            GD_matrix[i,:] = GD  
            y_hyp.append(self.y_train[np.argmin(GD)])
            
        return y_hyp, GD_matrix 
    
    
    #Trains the clifford classifier by preprocessing all examples in the training set
    def fit(self, X_train, y_train):      
        self.N = X_train.shape[0] #N is the number of training examples in the train set
        self.D = X_train.shape[1] #D is the time series dimension
        self.M = X_train.shape[2] #M is the length of the time series 
        self.D_shape = (special.binom(self.D,2)+1).astype(int) #D_shape is the dimension of the "space of XS" associated to a D-dimensional time series
        self.D_shape2 = (special.binom(self.D_shape,2)+1).astype(int) #D_shape2 is the dimension of the "space of XSS" associated to a D-dimensional time series

        if self.resample_length is None:
            self.X_train = X_train
        elif self.resample_length > 0:
            self.X_train = np.zeros((self.N, self.D, self.resample_length))
            for i in range(self.N):
                self.X_train[i,:,:] = self.__uniform_interp(X_train[i,:,:], self.resample_length)
        else:
            print("Please insert a valid value for resample_length")

        self.y_train = y_train
        self.XS_train, self.XSS_train = self.__clifford_preprocess(self.dist, self.X_train, self.shape_computation) #preprocess all training examples
    
    #Classify all candidates
    def predict(self, X_test):
        if self.resample_length is None:
            self.X_test = X_test
        elif self.resample_length > 0:
            self.X_test = np.zeros((X_test.shape[0], self.D, self.resample_length))
            for i in range(X_test.shape[0]):
                self.X_test[i,:,:] = self.__uniform_interp(X_test[i,:,:], self.resample_length)
        else:
            print("Please insert a valid value for resample_length")

        XS_test, XSS_test = self.__clifford_preprocess(self.dist, self.X_test, self.shape_computation) #preprocess candidate gestures
        y_hyp, GD_matrix = self.__compute_GD(self.X_test,self.X_train, XS_test, self.XS_train, XSS_test, self.XSS_train)

        return y_hyp, GD_matrix