from csv import reader
import numpy as np 
from plane_extraction import sort_paths
import copy
import pickle

folder = '../data/2012-04-29/'
groundtruth_dict = {}
gt = np.loadtxt(folder+'groundtruth_2012-04-29.csv', delimiter = ",")
for f in sort_paths(folder+'velodyne_sync/'):
    timestamp = int(f)
    all_difs = np.abs(gt[:,0]-timestamp)
    idx = np.argmin(all_difs)
    groundtruth_dict[f] = gt[idx,1:]
with open(folder+'groundtruth_dict.pickle', 'wb') as f:
    pickle.dump(groundtruth_dict, f)

