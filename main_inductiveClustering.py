import pandas as pd
import numpy as np
import pickle
import os
import time
import argparse

# Plotting Packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
savefig_options = dict(format="png", dpi=300, bbox_inches="tight")
plt.rcParams.update({'font.size': 14})

# Computation packages
from dtw import *
from kneed import KneeLocator

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

#Scipy
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import distance

from math import log, exp, tan, cos, pi, atan, ceil

import utils
import inductiveClustering

def main(args):
    
    filename = 'datasetInfo_AIS_Custom_01122021_31122021_CarFisHigMilPasPleSaiTan_600_43200_120'
    print(filename)
    data, data_pre, params, _, _ = utils.makeDataset('Unlabelled Data' + filename)
    data_test, data_pre_test, params_test, _, _ = utils.makeDataset('Labelled Data/datasetInfo_AIS_Custom_13122021_13122021_CarFisHigMilPasPleSaiTan_600_43200_120')

    #Define global variables
    lat_min, lat_max, lon_min, lon_max = utils.getPositionalBoundaries(params['binedges'], zoom=8)
    img = mpimg.imread('plots/historicBorn.png')
    
    #Calculate distances
    try:
        with open('data/' + filename +  'PositionalDeltaDist.pkl', "rb") as f:
            pos_cdist = pickle.load(f)
    except:
        pos_cdist = utils.cdistOfDelta(data_pre, data_pre, computeFull=False) #For spherical positional has to be 2D with columns [lon, lat]

        with open('data/' + filename +  'PositionalDeltaDist.pkl', "wb") as f:
            pickle.dump(pos_cdist, f, protocol=4)

    #try:
    #    with open('data/' + filename +  'KinematicDist.pkl', "rb") as f:
    #        kin_cdist = pickle.load(f)
    #except:
    #    kin_cdist = utils.cdistOfSpeedAndCourse(data_pre[:n], data_pre[:n], computeFull=False) #For spherical positional has to be 2D with columns [lon, lat]
    #
    #    with open('data/' + filename +  'KinematicDist.pkl', "wb") as f:
    #        pickle.dump(kin_cdist, f, protocol=4)

    if args.printSilhouetteScores:
        dist_range = np.arange(5,20,0.5)
        linkages = ["average"]
        all_silhouette_score = []
        all_n_clusters = []
        for linkage in linkages:
            silhouette_score = []
            n_clusters = []
            for dist_threshold in dist_range:
                model_tresholdDist = AgglomerativeClustering(
                    affinity='precomputed',
                    linkage= linkage, #‘ward’, ‘complete’, ‘average’, ‘single’ documents claim average linkage is best for non-euclidean affinites
                    compute_full_tree=True,    
                    distance_threshold=dist_threshold, 
                    n_clusters=None
                )
                model_tresholdDist = model_tresholdDist.fit(pos_cdist)
                
                n_clusters.append(model_tresholdDist.n_clusters_)
                try:
                    silhouette_score.append(metrics.silhouette_score(pos_cdist, model_tresholdDist.labels_, metric='precomputed'))
                except:
                    silhouette_score.append(-1)            
                    
            all_n_clusters.append(n_clusters)
            all_silhouette_score.append(silhouette_score)

        print(dist_range[np.argmax(all_silhouette_score[0])])    

        plt.rcParams.update({'font.size': 12})
        fig, ax1 = plt.subplots(1,1,figsize=(10,10))
        for i in range(len(all_silhouette_score)):
            ax1.plot(dist_range, all_silhouette_score[i], label='Silhouette Score')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        for i in range(len(all_n_clusters)):
            ax2.plot(dist_range, all_n_clusters[i], '--', label='Number of Clusters')

        plt.legend()
        ax1.set_title('Silhouette Score and Number of Clusters')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_xlabel('Distance Threshold in Hierarchical Clustering')
        ax2.set_ylabel('Number of clusters')  # we already handled the x-label with ax1
        plt.savefig("plots/" + filename + "PositionalSilhoutteScores.png", bbox_inches='tight',pad_inches = 0)
        plt.close()

    distanceThreshold=9.25

    try:
        with open('models/' + filename +  '_PositionalClusterer.pkl','rb') as f:
            pos_clusterer = pickle.load(f)
    except:
        pos_clusterer = AgglomerativeClustering(
            affinity='precomputed',
            linkage= 'average',
            compute_full_tree=True,    
            distance_threshold=distanceThreshold, 
            n_clusters=None
        )
        pos_clusterer = pos_clusterer.fit(pos_cdist)

        plt.rcParams.update({'font.size': 12})
        plt.figure(figsize=(30,10))
        plt.title("Hierarchical Clustering Dendrogram")
        # plot the top three levels of the dendrogram
        utils.plot_dendrogram(pos_clusterer, truncate_mode="lastp", p=len(np.unique(pos_clusterer.labels_)))
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.savefig('plots/' + filename + 'dendrogram.png')
        plt.close()
        
        with open('models/' + filename +  '_PositionalClusterer.pkl','wb') as f:
            pickle.dump(pos_clusterer, f, protocol=4)

    if args.printMDSofPosDistance:
        palette = sns.color_palette("Spectral", pos_clusterer.n_clusters_)
        colors = np.array([palette[i] for i in pos_clusterer.labels_])
        plt.rcParams.update({'font.size': 12})    
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        cdist_embedded = MDS(n_components=2, dissimilarity='precomputed').fit_transform(pos_cdist)
        ax.scatter(cdist_embedded[:,0], cdist_embedded[:,1],s=15,c=colors)
        plt.savefig('plots/' + filename + 'MDS.png')
        plt.close()

    if args.printPositionalClusters:
        plt.rcParams.update({'font.size': 12})
                
        for label in np.unique(pos_clusterer.labels_):
            indicies = np.where(pos_clusterer.labels_==label)[0]
            
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max])
            utils.forceAspect(ax,aspect=1)
            ax.set_xlim([lon_min, lon_max])
            ax.set_ylim([lat_min, lat_max])
            
            for index in indicies:
                utils.plotTrack(data[index], data[index][:,3], ax, color='k', insertSpeed=False)
            
            plt.savefig("plots/" + filename + "PositionalCluster" + str(label) + ".png", bbox_inches='tight',pad_inches = 0)
            plt.close()

    if args.computeKinematicDistances:
        for label in np.unique(pos_clusterer.labels_):
            indicies = np.where(pos_clusterer.labels_==label)[0]
            sample = [data_pre[i] for i in indicies]
            cdist = utils.cdistOfSpeedAndCourse(sample, sample, computeFull=False) #For spherical positional has to be 2D with columns [lon, lat]
    
            output = {
                'cdist': cdist,
                'indicies': indicies
            }
            
            with open('data/' + filename +  '_KinDistCluster' + str(label) + '.pkl', "wb") as f:
                pickle.dump(output, f, protocol=4)
    
    if args.makeInductiveClustering:
        classifier = KNeighborsClassifier(
            n_neighbors=3,
            metric = 'precomputed'
        )

        inductive_learner, k, errors = inductiveClustering.InductiveClusterer(pos_clusterer, classifier, filename, data_pre).fit(pos_cdist, pos_clusterer.labels_)
        outliers, score_samples, y_preds = inductive_learner.predict(data_pre_test, n_cores=None)
        
        output = {
            'outliers': outliers,
            'score_samples': score_samples,
            'y_preds': y_preds
        }
            
        with open('data/' + filename +  '_outlierData.pkl', "wb") as f:
            pickle.dump(output, f, protocol=4)

def parse_arguments(argv):
    """Command line parser.
    Use like:
    python main.py --arg1 string --arg2 value --arg4
    
    For help:
    python main.py -h
    """
    parser = argparse.ArgumentParser()
    
    #parser.add_argument('--arg1', type=str, default='String', help='String value. Default = "String"')
    #parser.add_argument('--arg2', type=int, default=50, choices=[50, 100, 200], help='Integer value with limited choices. Default = 50')
    #parser.add_argument('--arg3', type=float, default=0.001, help='Float value. Default = 0.001')
    #parser.add_argument('--arg4', type=bool, default=False, help='Bool value. Default = False')
    #parser.add_argument("--optional", action="store_true", help="Optional argument")
    
    parser.add_argument("--printSilhouetteScores", action="store_true", help="Make Silhouette Scores plots")
    parser.add_argument("--Compute1stageClusteringDistances", action="store_true", help="Compute combined distance matrix for 1 stage clustering")         
    parser.add_argument("--printPositionalClusters", action="store_true", help="Print the trajectories in every positional cluster")                  
    parser.add_argument("--computeKinematicDistances", action="store_true", help="Compute kinematic distance for every positional cluster for 2 stage clustering")                  
    parser.add_argument("--printMDSofPosDistance", action="store_true", help="Print the MDS of the positional distance matric color by cluster")                  
    parser.add_argument("--makeInductiveClustering", action="store_true", help="Make Inductive Clustering of Testset")
         
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
