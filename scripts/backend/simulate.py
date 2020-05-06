import numpy as np
from SLAMBackend import SLAMBackend
from geometry_utils import transform_plane_to_world, transform_plane_to_local, rot3D, \
                           normalize_planes, computeH, compute_world_pose
import copy
from grapher import show_trajectory, show_planes, show_trajectory_and_planes
import math
import matplotlib.pyplot as plt


def measurement_noise(std):
    return np.random.normal(0,std, 4)

def add_plane_noise(planes, std):
    normalized_planes = normalize_planes((planes + measurement_noise(std)).reshape(1, -1))
    return normalized_planes.reshape(-1,)

if __name__ == "__main__":
    np.random.seed(0)

    num_poses = 15
    odom_list = np.zeros((num_poses, 3))
    odom_list[:, :2] = 1
    odom_list[:,-1] = np.deg2rad(30)
    init_pose = np.array([0,0,0])

    gt_odom = copy.deepcopy(odom_list)

    odom_list += np.hstack([np.random.normal(0, 0.3, (num_poses, 2)), np.random.normal(0, 0.1, (num_poses, 1))])

    gt = [init_pose]
    for i,odom in enumerate(gt_odom):
        gt.append(compute_world_pose(gt[-1], odom))
    gt = np.vstack([gt])

    x = [init_pose]
    for i,odom in enumerate(odom_list):
        x.append(compute_world_pose(x[-1], odom))
    x = np.vstack([x])


    std_x = np.array([0.3, 0.3, .01]) # x, y, theta
    std_l = np.array([0.01, 0.01, 0.01, 0.005]) # nx, ny, nz, d
    std_p = np.array([0.05, 0.05, 0.001])

    plane1 = np.array([0.707, 0.707, 0, 5])
    plane2 = np.array([-0.707, 0.707, 0, 2])
    plane3 = np.array([2, 1, 0, 2])
    # planes_gt = normalize_planes(np.vstack([plane1, plane2, plane3]))

    num_planes = 20
    planes_gt = normalize_planes(np.hstack([np.random.randint(2, 10, (num_planes,2)), np.zeros((num_planes, 1)), np.random.randint(50, 100, (num_planes,1))]))
    print(planes_gt)
    obj = SLAMBackend(std_p, std_x, std_l, init_pose)

    plane_map = {i:-1 for i in range(num_planes)}
    count = 0
    n = 0 
    for i,odom in enumerate(odom_list):
        print("Iteration:", i)
        obj.add_pose_measurement(odom)

        # see all planes always
        landmark_measurements = np.vstack([np.hstack([transform_plane_to_local(gt[i],
                                                      add_plane_noise(plane, std_l)), i, j])
                                                      for j, plane in enumerate(planes_gt)])

        if count == 5:
            n +=1
            count=0
        # obj.add_landmark_measurement(landmark_measurements[n:n+10,:])
        # obj.add_landmark_measurement(landmark_measurements[n:(n+1)*5,:])
        obj.add_landmark_measurement(landmark_measurements)

        print(obj.s_l.shape)
        count +=1
        # import pdb; pdb.set_trace()
        obj.solve()

    show_trajectory(x, obj.s_x, gt)
    #obj.s_l,  planes_gt[:15,:], np.ones(planes_gt[:15,:].shape[0]))



