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

# Computation packages
from dtw import *
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn import metrics
from sklearn.neighbors import LocalOutlierFactor
from scipy.cluster.hierarchy import dendrogram
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from scipy.spatial import distance
from multiprocessing import Pool

from math import log, exp, tan, cos, pi, atan, ceil

def findcenters(edges):
    lat_edges, lon_edges, speed_edges, course_edges = edges
    
    lat_dim = len(lat_edges) - 1
    lon_dim = len(lon_edges) - 1
    speed_dim = len(speed_edges) - 1
    course_dim = len(course_edges) - 1
    
    lat_centers = [round((lat_edges[i]+lat_edges[i+1])/2,3) for i in range(len(lat_edges)-1)] 
    lon_centers = [round((lon_edges[i]+lon_edges[i+1])/2,3) for i in range(len(lon_edges)-1)] 
    speed_centers = [round((speed_edges[i]+speed_edges[i+1])/2,3) for i in range(len(speed_edges)-1)] 
    course_centers = [round((course_edges[i]+course_edges[i+1])/2,3) for i in range(len(course_edges)-1)]
    
    return lat_centers,lon_centers,speed_centers,course_centers

def get_static_map_bounds(lat, lng, zoom, sx, sy):
    # lat, lng - center
    # sx, sy - map size in pixels

    # 256 pixels - initial map size for zoom factor 0
    sz = 256 * 2 ** zoom

    #resolution in degrees per pixel
    res_lat = cos(lat * pi / 180.) * 360. / sz
    res_lng = 360./sz

    d_lat = res_lat * sy / 2
    d_lng = res_lng * sx / 2

    return ((lat-d_lat, lng-d_lng), (lat+d_lat, lng+d_lng))

def getPositionalBoundaries(edges, zoom=8):
    
    lat_centers, lon_centers, speed_centers, course_centers = findcenters(edges)

    lat_center = lat_centers[int(len(lat_centers) / 2)] 
    lon_center = lon_centers[int(len(lon_centers) / 2)] 

    SW_corner, NE_corner = get_static_map_bounds(lat_center, lon_center, zoom, 640, 640)
    lat_min, lon_min = SW_corner
    lat_max, lon_max = NE_corner
    
    return lat_min, lat_max, lon_min, lon_max

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def plotTrack(data, speed, ax, color=None, lsty='solid', insertSpeed=False, alpha=1):

    seq_len = data.shape[0]

    lat = data[:,1]
    lon = data[:,0]

    points = np.array([lon, lat]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    cmap=plt.get_cmap('inferno') #Black is start, yellow is end
    if color is None:
        colors=[cmap(float(ii)/(seq_len-1)) for ii in range(seq_len-1)]  
    else:
        colors = [color]*(seq_len-1)
        
    dashes = (1,0) if lsty=='solid' else (0.5,100)
    for ii in range(0,seq_len-1):
        segii=segments[ii]
        lii, = ax.plot(segii[:,0],segii[:,1],color=colors[ii],linestyle=lsty, dashes=dashes, zorder=1, alpha=alpha)#, linewidth=5
            
        lii.set_solid_capstyle('round')
    
    if color is not None:
        ax.scatter(lon[0],lat[0],color='k', zorder=2)
    
    if insertSpeed:
        ins = ax.inset_axes([0.05,0.6,0.35,0.4])
        ins.plot(speed)
        #ins.set_xlabel('Time', fontsize=20)
        #ins.set_ylabel('Speed', fontsize=20)
    
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    
    return ax

def plotStartPoint(data, ax):

    lat = data[:,1]
    lon = data[:,0]

    ax.scatter(lon[0],lat[0],color='k',)
        
    return ax

def convertCourseToTrig(course):

    trig = np.concatenate([np.expand_dims(np.sin(np.deg2rad(course)),axis=1), np.expand_dims(np.cos(np.deg2rad(course)),axis=1)],axis=1)
    
    return trig
    
def convertTrigToCourse(Trig):
    
    ### Trig[:,0] = sin, Trig[:,1] = cos
    
    course = np.rad2deg(np.arctan2(Trig[:,0], Trig[:,1]))
    course[course < 0] = course[course < 0]+360
    
    return course

def dist(p1, p2, p0): #Distance fram p0 to line segment p1-p2
    return np.linalg.norm(np.cross(p2-p1, p1-p0))/np.linalg.norm(p2-p1)

def pls_pos(track, epsilon=0.005, mu=2):
    d_max = 0
    i_max = 0
    
    begin = np.hstack([track[0,:2],np.array(0)*mu])
    end = np.hstack([track[-1,:2],np.array(track.shape[0]-1)*mu])
    for i in range(1,track.shape[0]):
        p0 = np.hstack([track[i,:2],np.array(i)*mu])
        d = dist(begin, end, p0)
        
        if d>=d_max:
            d_max = d
            i_max = i
            
    if d_max >= epsilon:
        A = pls_pos(track[:(i_max+1),:], epsilon)
        B = pls_pos(track[i_max:,:], epsilon)
        Tc = np.vstack([A, B[1:,:]])
    else:
        Tc = np.vstack([track[0,:], track[-1,:]])
        
    return Tc

def pls_speed(track, epsilon=0.5):
    d_max = 0
    i_max = 0
    
    for i in range(1,track.shape[0]):
        
        v = numpy.interp(np.array(i), np.array([0, track.shape[0]-1]), np.array([track[0,2], track[-1,2]]))
        d = np.sqrt((track[i,2]-v)**2)
        
        if d>=d_max:
            d_max = d
            i_max = i
            
    if d_max >= epsilon:
        A = pls_speed(track[:(i_max+1),:], epsilon)
        B = pls_speed(track[i_max:,:], epsilon)
        Tc = np.hstack([A, A[-1] + B[1:]])
    else:
        Tc = np.array([0, track.shape[0]-1])
        
    return Tc
    
def pls_2stage(track, epsilon=0.005, epsilon_v=0.1, mu=2):

    speed_shifts = pls_speed(track, epsilon_v)
    t = len(speed_shifts)-1
    
    Tc = np.empty((0, track.shape[1]))
    for i in range(t):
        begin = speed_shifts[i]
        end = speed_shifts[i+1]
        B = pls_pos(track[begin:(end+1),:], epsilon)
    
        Tc = np.vstack([Tc[:-1,:], B])
    
    return Tc

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
def getChildren(children, node, n_samples_):
    
    if node[0] < n_samples_:
        left = [node[0]]
    else:
        tmp = getChildren(children, children[node[0]-n_samples_], n_samples_)
        tmp[0].extend(tmp[1])
        left = tmp[0]
    
    if node[1] < n_samples_:
        right = [node[1]]
    else:
        tmp = getChildren(children, children[node[1]-n_samples_], n_samples_)
        tmp[0].extend(tmp[1])
        right = tmp[0]
    
    return left, right

def computeSpeedDistance(series1, series2):
    try:
        speed_dist = dtw(series1[:,2], series2[:,2], dist_method='seuclidean').distance
    except:
        speed_dist = 100
    
    return speed_dist

def computeCourseDistance(series1, series2):
    #arccos(1 - DTW of sine-cosine series using cosince distance / pi)   
    
    try:
        course_dist = dtw(series1[:,-1], series2[:,-1], dist_method=lambda u, v: (np.abs(u-v) if np.abs(u-v)<np.pi else np.pi-(np.abs(u-v) % np.pi))/np.pi).distance 
        #course_dist = dtw(series1[:,-2:], series2[:,-2:], dist_method=lambda u, v: np.arccos(1-distance.cosine(u, v))/pi).distance
    except:
        course_dist = 100 
    
    return course_dist


def computeDistance(series1, series2):

    #DTW of speed series using standardized Euclidean
    speed_dist = computeSpeedDistance(series1, series2)

    #Compute pairwise course distances of series1 and series2   
    
    course_dist = computeCourseDistance(series1, series2)
           
    #distance between 2 timeseries
    return speed_dist+course_dist

def cdistOfSpeedAndCourse(list_of_series1, list_of_series2, computeFull=True):
    
    list_of_series1 = [pls_2stage(track) for track in list_of_series1]
    list_of_series2 = [pls_2stage(track) for track in list_of_series2]
    
    list_of_series1 = [np.hstack([track[:,:3], np.deg2rad(convertTrigToCourse(track[:,-2:]).reshape((-1,1)))]) for track in list_of_series1] #uncomment if using the cosine distance 
    list_of_series2 = [np.hstack([track[:,:3], np.deg2rad(convertTrigToCourse(track[:,-2:]).reshape((-1,1)))]) for track in list_of_series2] #uncomment if using the cosine distance 
        
    n = len(list_of_series1)
    m = len(list_of_series2)
    res = np.zeros((n, m), dtype=np.float32)
    
    durations = []
    if computeFull:
        for i in range(n):
            for j in range(m):
                res[i,j] = computeDistance(list_of_series1[i], list_of_series2[j])
    else:
        for i in range(n):
            for j in range(i+1,m):
                start = time.time()
                res[i,j] = computeDistance(list_of_series1[i], list_of_series2[j])
                end = time.time()
                durations.append(end-start)
    
        res = res + res.T - np.diag(np.diag(res)) #Copy to lower triangle
     
    return res, durations

def Haver(x,y,deg=True):

    if deg:
        x,y = (x*np.pi/180.)%(2*np.pi),(y*np.pi/180.)%(2*np.pi)
    a = np.sin( np.abs(y[1]-x[1])/2. )**2 + np.cos(x[1]) * np.cos(y[1]) * np.sin( np.abs(y[0]-x[0])/2. )**2
    d = 2. * np.arctan( np.sqrt(a/(1-a)) ) * 6371.
    return d

def computeDeltaKL(D1,D2,wd=1.,wt=1.,dist=Haver):
    L = min(len(D1),len(D2))
    xL= max(len(D1),len(D2))
    
    d = map( dist ,D1[:L],D2[:L])
    d = np.array(list(d))
    
    return np.log(xL) - np.log(L) + wt*(d[:-1]+d[1:]).sum()/2./L

def computeDelta(D1,D2,dist=Haver):
    L = min(len(D1),len(D2))
    
    d = map( dist ,D1[:L],D2[:L])
    d = np.array(list(d))
    
    return (d[:-1]+d[1:]).sum()/2./L

def cdistOfDeltaKL(list_of_series1, list_of_series2, computeFull=True):
    
    n = len(list_of_series1)
    m = len(list_of_series2)
    res = np.zeros((n, m), dtype=np.float32)
    
    if computeFull:
        for i in range(n):
            for j in range(m):
                res[i,j] = computeDeltaKL(list_of_series1[i][:,:2], list_of_series2[j][:,:2])
    else:
        for i in range(n):
            for j in range(i+1,m):
                res[i,j] = computeDeltaKL(list_of_series1[i][:,:2], list_of_series2[j][:,:2])    
    
        res = res + res.T - np.diag(np.diag(res)) #Copy to lower triangle
    
    return res

def computeDeltaKinematics(D1,D2,wd=1.,wt=1.):
    def deltaKinematics(x,y):
        speed = np.abs(x[0]-y[0])
        speed = speed/V
        course = (np.abs(x[1]-y[1]) if np.abs(x[1]-y[1])<np.pi else np.pi-(np.abs(x[1]-y[1]) % np.pi))/np.pi
        
        return speed + course
    
    L = min(len(D1),len(D2))
    
    V = np.std(np.hstack([D1[:L,0],D2[:L,0]]))
    
    d = map( deltaKinematics ,D1[:L],D2[:L])
    d = np.array(list(d))
    
    return (d[:-1]+d[1:]).sum()/2./L


def cdistOfDeltaKinematics(list_of_series1, list_of_series2, computeFull=True):
    
    n = len(list_of_series1)
    m = len(list_of_series2)
    res = np.zeros((n, m), dtype=np.float32)
    
    if computeFull:
        for i in range(n):
            for j in range(m):
                res[i,j] = computeDeltaKinematics(list_of_series1[i][:,2:], list_of_series2[j][:,2:])
    else:
        for i in range(n):
            for j in range(i+1,m):
                res[i,j] = computeDeltaKinematics(list_of_series1[i][:,2:], list_of_series2[j][:,2:])    
    
        res = res + res.T - np.diag(np.diag(res)) #Copy to lower triangle
    
    return res
        
def cdistOfDelta(list_of_series1, list_of_series2, computeFull=True):
    
    n = len(list_of_series1)
    m = len(list_of_series2)
    res = np.zeros((n, m), dtype=np.float32)
    
    if computeFull:
        for i in range(n):
            for j in range(m):
                res[i,j] = computeDelta(list_of_series1[i][:,:2], list_of_series2[j][:,:2])
    else:
        for i in range(n):
            for j in range(i+1,m):
                res[i,j] = computeDelta(list_of_series1[i][:,:2], list_of_series2[j][:,:2])    
    
        res = res + res.T - np.diag(np.diag(res)) #Copy to lower triangle
    
    return res
        
def makeDataset(filename):

    #make dataset
    with open('data/' + filename + '.pkl', "rb") as f:
        params = pickle.load(f)

    datapath = params['dataFileName']
    indicies = params['indicies']

    N = len(indicies)
    seq_len = 360 #Make something for max length

    data = []
    data_pre = []
    mmsis = []
    shiptypes = []
    for i, index in enumerate(indicies):
        with open(datapath, 'rb') as file:
            file.seek(index)
            track = pickle.load(file)
            
        tmpdf = pd.DataFrame(track)
        tmpdf['course'] = tmpdf['course'].fillna(value=0)
        
        data_tmp = np.array(tmpdf[['lon','lat','speed','course']].values)
        data_tmp_pre = np.concatenate([data_tmp[:,:3], convertCourseToTrig(data_tmp[:,3])], axis=1)
        
        data.append(data_tmp)
        data_pre.append(data_tmp_pre)
        mmsis.append(track['mmsi'])
        shiptypes.append(track['shiptype'])
        
    return data, data_pre, params, mmsis, shiptypes

def minSamples_wrapper(eps, cdist):
    n = cdist.shape[0]
    return np.sum(cdist<eps)/n

def clustering_wrapper(eps, minSamples, cdist):
    clustering = DBSCAN(eps=eps, min_samples=int(minSamples),metric='precomputed').fit(cdist)
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #n_noise_ = list(labels).count(-1)

    return n_clusters_
        
def findNumClustersDBSCAN(cdist, numMaxEps = 500, epsilon_candidates=None, minLns_candidates=None):
        
    if epsilon_candidates is None:
        sorted_ = np.sort(cdist,axis=1)
        epsilon_candidates_all = np.mean(sorted_,axis=0)
    
    epsilon_candidates = epsilon_candidates_all[:numMaxEps] if len(epsilon_candidates_all)>numMaxEps else epsilon_candidates_all
    if minLns_candidates is None:
        minLns_candidates = np.zeros((len(epsilon_candidates)))

        pool = Pool(processes=8)
        results = [pool.apply_async(minSamples_wrapper, [eps, cdist]) for eps in epsilon_candidates]
        for idx, val in enumerate(results):
            minLns_candidates[idx] = val.get()
        pool.close()

    #Make clustering
    num_clusters = np.zeros((numMaxEps-1))

    #Parallel was found to be slower
    #pool = Pool(processes=8)
    #results = [pool.apply_async(clustering_wrapper, [eps, minSamples, cdist]) for eps, minSamples in zip(epsilon_candidates[1:], minLns_candidates[1:])]
    #for idx, val in enumerate(results):
    #    num_clusters[idx] = val.get()
    #pool.close()   
    
    for idx, (eps, minSamples) in enumerate(zip(epsilon_candidates[1:], minLns_candidates[1:])):
        num_clusters[idx] = clustering_wrapper(eps, minSamples, cdist)    
     
    return epsilon_candidates, minLns_candidates, num_clusters
