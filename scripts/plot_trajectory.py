import numpy as np
import pickle
from plane_extraction import sort_paths
import matplotlib.pyplot as plt

def plot_trajectory(gt_dict_directory, trajcetory_list):
    gt_dict = pickle.load( open(gt_dict_directory, "rb" ) )
    original_transform = gt_dict[trajectory_list[0][0]]
    theta_original =original_transform[-1]#-np.deg2rad(90.703)
    print(original_transform[-3:])
    x_original = original_transform[1]
    y_original = original_transform[2]
    R = np.array([[np.cos(theta_original),-np.sin(theta_original)],[np.sin(theta_original),np.cos(theta_original)]])
    print("Total time traveled:",(trajcetory_list[-1][0]-trajcetory_list[0][0])*1e-6)
    trajectory = np.zeros((1,2))
    gt_trajectory = np.zeros((0,2))
    for timestamp,transformation in trajectory_list:
        trajectory = np.vstack((trajectory,np.array([transformation[0,-1],transformation[1,-1]])))
        gt_transform = gt_dict[timestamp]
        transformed_gt_xy = np.array([gt_transform[1]-x_original,gt_transform[2]-y_original])
        transformed_gt_xy = np.linalg.inv(R)@transformed_gt_xy.reshape((2,1))
        gt_trajectory = np.vstack((gt_trajectory,np.array([transformed_gt_xy[0,0],transformed_gt_xy[1,0]])))
    
    RMSE = np.sqrt(np.mean((trajectory[:-1,:2]-gt_trajectory[:,:2])**2))
    print(RMSE)
    plt.plot(trajectory[:,0], trajectory[:,1],'r--',label="trajectory")
    plt.plot(gt_trajectory[:,0],gt_trajectory[:,1],'b',label="ground truth")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("Lidar SLAM VS Ground Truth Performance")
    plt.xlabel("x in meters")
    plt.ylabel("y in meters")
    plt.show()

if __name__ == '__main__':
    folder = '../data/2012-04-29/'
    # traj = np.zeros((0,4))
    # for f in sort_paths(folder+'velodyne_sync/'):
    #     timestamp = int(f)
    #     traj = np.vstack((traj,np.array([timestamp,0,0,0])))

    trajectory_list = pickle.load( open('./trajectory_long.pickle', "rb" ) )
    plot_trajectory(folder+'groundtruth_dict.pickle',trajectory_list)