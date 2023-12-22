import pandas as pd
import numpy as np
import pickle
import os
import time

# Plotting Packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
savefig_options = dict(format="png", dpi=300, bbox_inches="tight")
plt.rcParams.update({'font.size': 14})

#Parallelization
import multiprocessing 
from joblib import Parallel, delayed

# Computation packages
from dtw import *

#Sklearn
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import LeaveOneOut    
from sklearn.model_selection import cross_val_predict

#Scipy
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import distance

from math import log, exp, tan, cos, pi, atan, ceil

from utils import utils


def _classifier_has(attr):
    """Check if we can delegate a method to the underlying classifier.

    First, we check the first fitted classifier if available, otherwise we
    check the unfitted classifier.
    """
    return lambda estimator: (
        hasattr(estimator.classifier_, attr)
        if hasattr(estimator, "classifier_")
        else hasattr(estimator.classifier, attr)
    )

class InductiveClusterer(BaseEstimator):
    def __init__(self, clusterer, classifier, filename, trainingTracks, n_neighbors=5, contamination=0.05, crossValClassifier=False, removeStationaryObservations=False):
        self.clusterer = clusterer
        self.classifier = classifier
        self.filename = filename
        self.trainingData = trainingTracks
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.removeStationaryObservations = removeStationaryObservations
        
        self.crossValClassifier = crossValClassifier
    
    def fit_positionalClassifier(self, X, y=None):
                
        errors = None
        k = self.classifier.get_params('n_neighbors')
        
        if y is None:
            y = self.clusterer.fit_predict(X)
        
        if self.crossValClassifier:
            n = X.shape[0]
            K = 10 # Number of folds

            #Create a vector of length n that contains equal amounts of numbers from 1 to K
            I = np.asarray([0] * n)
            for i in range(n):
                I[i] = (i) % K + 1
            #Permute that vector. 
            I = I[np.random.permutation(n)]  
            
            k_range = range(1, 101)
            errors = np.zeros((K, len(k_range)))
            for i in k_range:  
                classifier = KNeighborsClassifier(n_neighbors=i, metric = 'precomputed')  
                
                count = 0
                for j in range(1, K+1): # for j in range(n):
                    #selector = [k for k in range(n) if k != j]
                    
                    XTrain = X[j != I, :][:,j != I] #X[selector,:][:,selector]
                    yTrain = y[j != I] #y[selector]
                    XTest = X[j == I, :][:,j != I] #np.expand_dims(X[j][selector],0)
                    yTest = y[j == I] #y[j]
                                    
                    classifier.fit(XTrain, yTrain)
                    y_pred = classifier.predict(XTest)
                    errors[j-1,i-1] = np.mean(y_pred != yTest)
                
            error = np.mean(errors, axis = 0)
            
            k = k_range[np.argmin(error)]
            print(f"The optimal value of k is found to be {k}")
            params = {'n_neighbors': k}
            self.classifier.set_params(**params)
        
        self.classifier.fit(X, y)
        
        return y, k, errors
                
    def fit(self, X, y=None):
        
        y, k, errors = self.fit_positionalClassifier(X, y)
        
        #For each cluster in self.clusterer train
        self.clusterOutlierDetection = dict()
        for pos_cluster in range(self.clusterer.n_clusters_):            
            with open('data/' + self.filename +  '_KinDistCluster' + str(pos_cluster) + '.pkl', "rb") as f:
                kin_cdist = pickle.load(f)

            clusteridx = kin_cdist['indicies']
            kin_cdist = kin_cdist['cdist']
            
            if self.removeStationaryObservations:
                avg_speeds = np.array([np.mean(self.trainingData[i][:,2]) for i in clusteridx])
                filter_ = (avg_speeds>0.3)

                clusteridx = clusteridx[filter_]
                kin_cdist = kin_cdist[filter_,:][:,filter_]
                            
            if len(clusteridx)>self.n_neighbors:
                self.clusterOutlierDetection[pos_cluster] = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination, metric='precomputed', novelty=True).fit(kin_cdist)
            else:
                self.clusterOutlierDetection[pos_cluster] = -1
        
        return self, k, errors

    def computePositionalDistances(self, listOfTracks, listOfTrainingData):
        positional_train = [data[:,:2] for data in listOfTrainingData]
        positional_test = [data[:,:2] for data in listOfTracks]
        cdist_new = utils.cdistOfDelta(positional_test, positional_train, computeFull=True)
            
        return cdist_new
    
    def computeKinematicDistances(self, listOfTracks, listOfTrainingData):
        kinematic_train = [data for data in listOfTrainingData]
        kinematic_test = [data for data in listOfTracks]        
        cdist_new = utils.cdistOfSpeedAndCourse(kinematic_test, kinematic_train, computeFull=True)
            
        return cdist_new    
    
    @available_if(_classifier_has("predict"))
    def predictPositionalCluster(self, X):
                
        cdist = self.computePositionalDistances(X, self.trainingData)
        return self.classifier.predict(cdist)
    
    @available_if(_classifier_has("predict"))
    def predict(self, X, n_cores = 1):                
        
        y_preds = self.predictPositionalCluster(X)
        
        n = len(X)
        outliers = np.zeros(n)
        score_samples = np.zeros(n)        

        if n_cores==1:
            #For each in X take correct subsample of self.data that fits y. 
            for y in np.unique(y_preds):
                
                inds = np.where(y_preds == y)[0]
                samples = [X[i] for i in inds]



                if len(samples) > 0:
                    #Find trainingData from sample cluster
                    with open('data/' + self.filename +  '_KinDistCluster' + str(y) + '.pkl', "rb") as f:
                        kin_cdist = pickle.load(f)

                    clusteridx = kin_cdist['indicies']
                    kin_cdist = kin_cdist['cdist']
                    
                    if self.removeStationaryObservations:
                        avg_speeds = np.array([np.mean(self.trainingData[i][:,2]) for i in clusteridx])
                        filter_ = (avg_speeds>0.3)

                        clusteridx = clusteridx[filter_]
                        kin_cdist = kin_cdist[filter_,:][:,filter_]
                    
                    if len(clusteridx)>self.n_neighbors:
                        cluster_data = [self.trainingData[idx] for idx in clusteridx]

                        #Compute kinematic distances
                        kin_cdist = self.computeKinematicDistances(samples, cluster_data)
                        #Use the outlier detection
                        outliers[inds] = self.clusterOutlierDetection[y].predict(kin_cdist)
                        score_samples[inds] = self.clusterOutlierDetection[y].score_samples(kin_cdist)          
                    else:
                        outliers[inds] = -1
                        score_samples[inds] = -999999    
                                    
                    #print(f'Track {} is predicted in cluster {y} and has outlier prediction {outliers} and score {score_samples}')
        else:
            n_cores = multiprocessing.cpu_count() if n_cores is None else n_cores
            
            def my_function(y):
                
                inds = np.where(y_preds == y)[0]
                samples = [X[i] for i in inds]

                if len(samples) > 0:
                    #Find trainingData from sample cluster
                    with open('data/' + self.filename +  '_KinDistCluster' + str(y) + '.pkl', "rb") as f:
                        kin_cdist = pickle.load(f)

                    clusteridx = kin_cdist['indicies']
                    kin_cdist = kin_cdist['cdist']
                    
                    if self.removeStationaryObservations:
                        avg_speeds = np.array([np.mean(self.trainingData[i][:,2]) for i in clusteridx])
                        filter_ = (avg_speeds>0.3)

                        clusteridx = clusteridx[filter_]
                        kin_cdist = kin_cdist[filter_,:][:,filter_]
                            
                    if len(clusteridx)>self.n_neighbors:
                        cluster_data = [self.trainingData[idx] for idx in clusteridx]

                        #Compute kinematic distances
                        kin_cdist = self.computeKinematicDistances(samples, cluster_data)

                        out = self.clusterOutlierDetection[y].predict(kin_cdist) 
                        score = self.clusterOutlierDetection[y].score_samples(kin_cdist)
                    else:
                        out = np.ones(len(samples)) * -1
                        score = np.ones(len(samples)) * -999999

                return inds, out, score

            processed_list = Parallel(n_jobs=n_cores)([delayed(my_function)(y) for y in np.unique(y_preds)])
            processed_list = list(map(np.concatenate, zip(*processed_list)))
            
            inds, is_outlier, scores = processed_list
        
            outliers[inds] = is_outlier
            score_samples[inds] = scores     
                
        return outliers, score_samples, y_preds

    
    @available_if(_classifier_has("decision_function"))
    def decision_function(self, X):
        check_is_fitted(self)
        return self.classifier.decision_function(X)
        
    
