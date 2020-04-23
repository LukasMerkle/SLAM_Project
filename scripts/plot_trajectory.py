import numpy as np
import pickle
from plane_extraction import sort_paths
import matplotlib.pyplot as plt

def plot_trajectory(gt_dict_directory, trajcetory):
    gt_dict = pickle.load( open(gt_dict_directory, "rb" ) )
    gt_traj = np.zeros((0,6))
    for pos in trajcetory:
        gt_traj = np.vstack((gt_traj,gt_dict[pos[0]]))
    plt.plot(gt_traj[:,0], gt_traj[:,1])
    plt.show()
if __name__ == '__main__':
    traj = np.zeros((0,4))
    folder = '../data/2012-04-29/'
    for f in sort_paths(folder+'velodyne_sync/'):
        timestamp = int(f)
        traj = np.vstack((traj,np.array([timestamp,0,0,0])))
    plot_trajectory(folder+'groundtruth_dict.pickle',traj)