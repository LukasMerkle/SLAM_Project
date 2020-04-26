import numpy as np
from SLAMBackend import SLAMBackend
from geometry_utils import transform_plane_to_world, transform_plane_to_local, rot3D, \
                           normalize_planes, computeH
import copy
from grapher import show_trajectory
import matplotlib.pyplot as plt
import math

def compute_world_pose(pos1, pos2):
    H1 = computeH(pos1)
    H2 = computeH(pos2)
    H = np.dot(H1, H2)
    t = math.atan2(H[1,0], H[0,0])
    x = H[0,-1]
    y = H[1,-1]
    return np.array([x,y,t])

if __name__ == "__main__":
    std_x = np.array([0.3, 0.3, 0.15]) # x, y, theta
    std_l = np.array([0.1, 0.1, 0.1, 0.1]) # nx, ny, nz, d
    std_p = np.array([0.05, 0.05, 0.001])
    init_pose = np.array([0,0,0])

    dic = np.load('planes_and_poses.npz')
    all_planes = dic['planes']
    odom_list = dic['poses']

    gt_dic = np.load('ground_truth.npz')
    gt = gt_dic['gt_trajectory']
    print(odom_list)
    exit()
    print(all_planes[:5, :].astype(int))
    obj = SLAMBackend(std_p, std_x, std_l, init_pose)


    pure_odometry = [init_pose]
    for i,odom in enumerate(odom_list):
        pure_odometry.append(compute_world_pose(pure_odometry[-1], odom))
    pure_odometry = np.vstack([pure_odometry])

    np.save('pure_odometry', pure_odometry)

    for i,odom in enumerate(odom_list):
        if (i % 10 == 0):
            print("Iteration:", i)
        if (i == 150):
            break
        obj.add_pose_measurement(odom)

        # see all planes always
        current_plane_inds = np.where(all_planes[:, obj.P_IDX] == i)[0]
        landmark_measurements = all_planes[current_plane_inds,:]

        obj.add_landmark_measurement(landmark_measurements)
        obj.solve()

    np.save('corrected', obj.s_x)
    s_x = np.load('corrected.npy')

    # show_trajectory(np.cumsum(odom_list[:40], axis=0), obj.s_x, odom_list[:40])
    plt.figure()
    plt.plot(pure_odometry[:,0], pure_odometry[:,1], label= "Odometry")
    plt.plot(s_x[:,0], s_x[:,1], label="Corrected")
    plt.plot(gt[:150,0], gt[:150,1], label="Ground Truth")
    plt.show()


