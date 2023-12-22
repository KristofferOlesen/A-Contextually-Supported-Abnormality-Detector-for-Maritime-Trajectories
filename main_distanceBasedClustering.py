import pandas as pd
import numpy as np
import pickle
import os
import sys
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
#import traj_dist.distance as tdist
from kneed import KneeLocator
from multiprocessing import Pool

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

#Scipy
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import distance

from math import log, exp, tan, cos, pi, atan, ceil

from utils import utils

def speedFunc_wrapper(tracks):
    return utils.computeSpeedDistance(tracks[0], tracks[1])        

def courseFunc_wrapper(tracks):
    return utils.computeCourseDistance(tracks[0], tracks[1])       

def dtwFunc_wrapper(tracks):
    return tdist.dtw(tracks[0][:,:2], tracks[1][:,:2], type_d="spherical")     

def hausdorffFunc_wrapper(tracks):
    return tdist.hausdorff(tracks[0][:,:2], tracks[1][:,:2], type_d="spherical")       

def main(args):
    
    filename = 'datasetInfo_AIS_Custom_01112021_30112021_CarDivFisHigMilOthPasPilPleSaiTan_600_43200_120'
    print(filename)
    data, data_pre, params, _, _ = utils.makeDataset('Unlabelled Data/' + filename)

    #Define global variables
    lat_min, lat_max, lon_min, lon_max = utils.getPositionalBoundaries(params['binedges'], zoom=8)
    img = mpimg.imread('plots/historicSjælland.png')
    
    
    #Compute Sample of kindistances
    if args.TimeKinematicDistanceMeasure:
    #Calculate distances
        n = 5000
        
        indicies = np.random.choice(len(data), size=n, replace=False)
        sample = [data_pre[i] for i in indicies]
                
        kin_cdist, durations = utils.cdistOfSpeedAndCourse(sample, sample, computeFull=False) #For spherical positional has to be 2D with columns [lon, lat]
        
        print(f'Number of pairs for which Kindist was calculated: {len(durations)}')
        print(f'Mean of calculation time: {np.mean(durations)}')
        print(f'Std of calculation time: {np.std(durations)}')
        
        exit()
    
    #Calculate positional distances
    try:
        with open('data/' + filename +  'PositionalDeltaDist.pkl', "rb") as f:
            pos_cdist = pickle.load(f)
    except:
        pos_cdist = utils.cdistOfDelta(data_pre, data_pre, computeFull=False) #For spherical positional has to be 2D with columns [lon, lat]

        with open('data/' + filename +  'PositionalDeltaDist.pkl', "wb") as f:
            pickle.dump(pos_cdist, f, protocol=4)
            
        exit()

    #Compress trajectories
    try:
        with open('data/' + filename +  '_CompressedTracks.pkl', "rb") as f:
            data_compressed = pickle.load(f)
    except:
        data_compressed = [utils.pls_2stage(track) for track in data_pre]
        
        with open('data/' + filename +  '_CompressedTracks.pkl', "wb") as f:
            pickle.dump(data_compressed, f, protocol=4)

    if args.ComputeSpeedcdist:  
        n = len(data_compressed)
        speed_cdist = np.zeros((n, n), dtype=np.float32)
            
        pool = Pool(processes=8)
        for i in range(n):
            speed_cdist[i,(i+1):] = pool.map(speedFunc_wrapper, [(data_compressed[i], data_compressed[j]) for j in range(i+1,n)])
        pool.close()
        speed_cdist = speed_cdist + speed_cdist.T - np.diag(np.diag(speed_cdist)) #Copy to lower triangle
        
        with open('data/' + filename +  '_Speedcdist.pkl', "wb") as f:
            pickle.dump(speed_cdist, f, protocol=4)

    if args.ComputeCoursecdist:
        n = len(data_compressed)        
        data_compressed_test = [np.hstack([track[:,:3], np.deg2rad(utils.convertTrigToCourse(track[:,-2:]).reshape((-1,1)))]) for track in data_compressed] #uncomment if using the cosine distance 

        split2 = n//2
        split1 = split2//2
        split3 = split1+split2
        print(split1)
        print(split2)
        print(split3)
        
        if args.section==1:
            course_cdist = np.zeros((split2, split2), dtype=np.float32)
            pool = Pool(processes=8)
            for i in range(split1):
                course_cdist[i,(i+1):split1] = pool.map(courseFunc_wrapper, [(data_compressed_test[i], data_compressed_test[j]) for j in range(i+1,split1)])
                print(i,flush=True)
            for i in range(split1, split2):
                course_cdist[i,(i+1):] = pool.map(courseFunc_wrapper, [(data_compressed_test[i], data_compressed_test[j]) for j in range(i+1,split2)])
                print(i,flush=True)
            pool.close()

            course_cdist = course_cdist + course_cdist.T - np.diag(np.diag(course_cdist)) #Copy to lower triangle                  
        
        elif args.section==2:
            course_cdist = np.zeros((split1, split2-split1), dtype=np.float32)
            pool = Pool(processes=8)
            for i in range(split1):
                course_cdist[i,:] = pool.map(courseFunc_wrapper, [(data_compressed_test[i], data_compressed_test[j]) for j in range(split1,split2)])
                print(i,flush=True)
            pool.close()
        
        elif args.section==3:
            course_cdist = np.zeros((split1, split3-split2), dtype=np.float32)
            pool = Pool(processes=8)
            for i in range(split1):
                course_cdist[i,:] = pool.map(courseFunc_wrapper, [(data_compressed_test[i], data_compressed_test[j]) for j in range(split2,split3)])
                print(i,flush=True)
            pool.close()
        
        elif args.section==4:
            course_cdist = np.zeros((split2-split1, split3-split2), dtype=np.float32)
            pool = Pool(processes=8)
            for i in range(split2-split1):
                course_cdist[i,:] = pool.map(courseFunc_wrapper, [(data_compressed_test[split1+i], data_compressed_test[j]) for j in range(split2,split3)])
                print(i,flush=True)
            pool.close()
        
        elif args.section==5:
            course_cdist = np.zeros((split1, n-split3), dtype=np.float32)
            pool = Pool(processes=8)
            for i in range(split1):
                course_cdist[i,:] = pool.map(courseFunc_wrapper, [(data_compressed_test[i], data_compressed_test[j]) for j in range(split3,n)])
                print(i,flush=True)
            pool.close()
        
        elif args.section==6:
            course_cdist = np.zeros((split2-split1, n-split3), dtype=np.float32)
            pool = Pool(processes=8)
            for i in range(split2-split1):
                course_cdist[i,:] = pool.map(courseFunc_wrapper, [(data_compressed_test[split1+i], data_compressed_test[j]) for j in range(split3,n)])
                print(i,flush=True)
            pool.close()
        
        elif args.section==7:
            course_cdist = np.zeros((split3-split2, n-split3), dtype=np.float32)
            pool = Pool(processes=8)
            for i in range(split3-split2):
                course_cdist[i,:] = pool.map(courseFunc_wrapper, [(data_compressed_test[split2+i], data_compressed_test[j]) for j in range(split3,n)])
                print(i,flush=True)
            pool.close()
        
        elif args.section==8:
            course_cdist = np.zeros((n-split2, n-split2), dtype=np.float32)
            pool = Pool(processes=8)
            for i in range(split3-split2):
                course_cdist[i,(i+1):(split3-split2)] = pool.map(courseFunc_wrapper, [(data_compressed_test[split2+i], data_compressed_test[j]) for j in range(split2+i+1,split3)])
                print(i,flush=True)
            for i in range(split3-split2,n-split2):
                course_cdist[i,(i+1):] = pool.map(courseFunc_wrapper, [(data_compressed_test[split2+i], data_compressed_test[j]) for j in range(split2+i+1,n)])
                print(i,flush=True)
            pool.close()

            course_cdist = course_cdist + course_cdist.T - np.diag(np.diag(course_cdist)) #Copy to lower triangle
                
        with open('data/' + filename +  '_Coursecdist_section' + str(args.section) + '.pkl', "wb") as f:
            pickle.dump(course_cdist, f, protocol=4)
    
    if args.ComputeDTWcdist:
        n = len(data_compressed)
        cdist = np.zeros((n, n), dtype=np.float32)
        
        pool = Pool(processes=8)
        for i in range(n):
            cdist[i,(i+1):] = pool.map(dtwFunc_wrapper, [(data_compressed[i], data_compressed[j]) for j in range(i+1,n)])
        pool.close()
        cdist = cdist + cdist.T - np.diag(np.diag(cdist)) #Copy to lower triangle
        
        with open('data/' + filename +  '_dtwcdist.pkl', "wb") as f:
            pickle.dump(cdist, f, protocol=4)
    
    if args.ComputeHausdorffcdist:
        n = len(data_compressed)
        cdist = np.zeros((n, n), dtype=np.float32)
        
        pool = Pool(processes=8)
        for i in range(n):
            cdist[i,(i+1):] = pool.map(hausdorffFunc_wrapper, [(data_compressed[i], data_compressed[j]) for j in range(i+1,n)])
        pool.close()
        cdist = cdist + cdist.T - np.diag(np.diag(cdist)) #Copy to lower triangle
        
        with open('data/' + filename +  '_hausdorffcdist.pkl', "wb") as f:
            pickle.dump(cdist, f, protocol=4)
    
    if args.combineCourseMatricesToCdist:
        n = len(data_compressed)
        course_cdist = np.zeros((n, n), dtype=np.float32)

        split2 = n//2
        split1 = split2//2
        split3 = split1+split2

        for section in range(1,9):
            with open('data/' + filename +  '_Coursecdist_section' + str(section) + '.pkl', "rb") as f:
                cdist = pickle.load(f)
            
            if section==1:
                course_cdist[:split2,:split2] = course_cdist[:split2,:split2] + cdist 
            if section==2:
                course_cdist[:split1,split1:split2] = course_cdist[:split1,split1:split2] + cdist 
            if section==3:
                course_cdist[:split1,split2:split3] = course_cdist[:split1,split2:split3] + cdist 
            if section==4:
                course_cdist[split1:split2,split2:split3] = course_cdist[split1:split2,split2:split3] + cdist 
            if section==5:
                course_cdist[:split1,split3:] = course_cdist[:split1,split3:] + cdist 
            if section==6:
                course_cdist[split1:split2,split3:] = course_cdist[split1:split2,split3:] + cdist 
            if section==7:
                course_cdist[split2:split3,split3:] = course_cdist[split2:split3,split3:] + cdist
            if section==8:
                course_cdist[split2:,split2:] = course_cdist[split2:,split2:] + cdist 
                
            X = np.triu(course_cdist)
            course_cdist = X + X.T - np.diag(np.diag(X))
            course_cdist
            
            with open('data/' + filename +  '_Coursecdist.pkl', "wb") as f:
                pickle.dump(course_cdist, f, protocol=4)
       
    if args.findNumClusters:
        distancemeasure = args.distancemeasure
        if distancemeasure=='hausdorff':
            filenamelocal = 'data/' + filename +  '_hausdorffcdist.pkl'
            with open(filenamelocal, "rb") as f:
                cdist = pickle.load(f)
        elif distancemeasure=='dtw':
            filenamelocal = 'data/' + filename +  '_dtwcdist.pkl'
            with open(filenamelocal, "rb") as f:
                cdist = pickle.load(f)
        elif distancemeasure=='latlon':
            filenamelocal = 'data/' + filename +  'PositionalDeltaDist.pkl'
            with open(filenamelocal, "rb") as f:
                cdist = pickle.load(f)
        elif distancemeasure=='AllDBSCAN':
            filenamelocal = 'data/' + filename +  'PositionalDeltaDist.pkl'
            with open(filenamelocal, "rb") as f:
                cdist = pickle.load(f)
            filenamelocal = 'data/' + filename +  '_Speedcdist.pkl'
            with open(filenamelocal, "rb") as f:
                cdist += pickle.load(f) #Add to previous cdist
            filenamelocal = 'data/' + filename +  '_Coursecdist.pkl'
            with open(filenamelocal, "rb") as f:
                cdist += pickle.load(f) #Add to previous cdist                                       

        epsilon_candidates, minLns_candidates, num_clusters = utils.findNumClustersDBSCAN(cdist, numMaxEps = 500, epsilon_candidates=None, minLns_candidates=None)
        out = {
            'epsilon_candidates': epsilon_candidates, 
            'minLns_candidates': minLns_candidates, 
            'num_clusters': num_clusters
        }
        with open('data/numClustersDBSCAN_' + distancemeasure + '.pkl', "wb") as f:
            pickle.dump(out, f, protocol=4)
    
    
    distanceThreshold=10

    if False:
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

    parser.add_argument('--distancemeasure', type=str, default='none', choices=['none', 'hausdorff', 'dtw', 'latlon', 'AllDBSCAN'], help='Distance measure to compute results for. Default = none')
    parser.add_argument('--section', type=int, default=0, help='Integer value to decide which section of course distance matrix to calculate. Default = 0')

    parser.add_argument("--ComputeSpeedcdist", action="store_true", help="Compute the speed distance matrix")                  
    parser.add_argument("--ComputeCoursecdist", action="store_true", help="Compute the course distance matrix")                  
    parser.add_argument("--ComputeDTWcdist", action="store_true", help="Compute the DTW distance matrix")                  
    parser.add_argument("--ComputeHausdorffcdist", action="store_true", help="Compute the Hausdorff distance matrix")                  
    parser.add_argument("--findNumClusters", action="store_true", help="Find num of cluster for DBSCAN settings")                  
    parser.add_argument("--printPositionalClusters", action="store_true", help="Print the trajectories in every positional cluster")                  
    parser.add_argument("--TimeKinematicDistanceMeasure", action="store_true", help="Time the kinematic distance measure")                  

    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
