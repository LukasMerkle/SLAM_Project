import numpy as np
import pickle
from plane_extraction import sort_paths
import matplotlib.pyplot as plt

def plot_trajectory(gt_dict_directory, trajcetory_list):
    gt_dict = pickle.load( open(gt_dict_directory, "rb" ) )
    original_transform = gt_dict[trajectory_list[0][0]]
    theta_original =original_transform[-1] +np.pi
    x_original = original_transform[0]
    y_original = original_transform[1]
    
    trajectory = np.zeros((0,2))
    gt_trajectory = np.zeros((0,2))
    for timestamp,transformation in trajectory_list:
        trajectory = np.vstack((trajectory,np.array([transformation[0,-1],transformation[1,-1]])))
        gt_transform = gt_dict[timestamp]
        transformed_gt_x = (gt_transform[0]-x_original)/np.cos(theta_original)
        transformed_gt_y = (gt_transform[1]-y_original)/np.sin(theta_original)
        gt_trajectory = np.vstack((gt_trajectory,np.array([transformed_gt_x,transformed_gt_y])))

    plt.plot(trajectory[:,0], trajectory[:,1],'r--',gt_trajectory[:,0],gt_trajectory[:,1],'b')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == '__main__':
    folder = '../data/2012-04-29/'
    # traj = np.zeros((0,4))
    # for f in sort_paths(folder+'velodyne_sync/'):
    #     timestamp = int(f)
    #     traj = np.vstack((traj,np.array([timestamp,0,0,0])))

    trajectory_list = pickle.load( open('./trajectory.pickle', "rb" ) )
    plot_trajectory(folder+'groundtruth_dict.pickle',trajectory_list)