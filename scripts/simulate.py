import numpy as np
from SLAMBackend import SLAMBackend
from geometry_utils import transform_plane_to_world, transform_plane_to_local, rot3D, \
                           normalize_planes
import copy
from grapher import show_trajectory, show_planes, show_trajectory_and_planes

def measurement_noise(std):
    return np.random.normal(0,std, 4)

def add_plane_noise(planes, std):
    normalized_planes = normalize_planes((planes + measurement_noise(std)).reshape(1, -1))
    return normalized_planes.reshape(-1, )


if __name__ == "__main__":
    num_poses = 15
    odom_list = np.zeros((num_poses, 3))
    odom_list[:, :2] = 1
    init_pose = np.array([0,0,0])

    gt_odom = copy.deepcopy(odom_list)
    gt = np.vstack([init_pose, np.cumsum(gt_odom, axis=0)])

    odom_list += np.hstack([np.random.normal(0, 0.3, (num_poses, 2)), np.random.normal(0, 0.1, (num_poses, 1))])

    x = np.cumsum(odom_list, axis=0)

    std_x = np.array([0.3, 0.3, .1]) # x, y, theta
    std_l = np.array([0.01, 0.01, 0.01, 0.005]) # nx, ny, nz, d
    std_p = np.array([0.05, 0.05, 0.001])

    plane1 = np.array([0.707, 0.707, 0, 5])
    plane2 = np.array([-0.707, 0.707, 0, 2])
    plane3 = np.array([2, 1, 0, 2])
    planes_gt = normalize_planes(np.vstack([plane1, plane2, plane3]))

    obj = SLAMBackend(std_p, std_x, std_l, init_pose)

    for i,odom in enumerate(odom_list):
        print("Iteration:", i)
        obj.add_pose_measurement(odom)

        # see all planes always
        landmark_measurements = np.vstack([np.hstack([transform_plane_to_local(gt[i],
                                                      add_plane_noise(plane, std_l)), j, i])
                                                      for j, plane in enumerate(planes_gt)])

        # measurement1 = np.hstack([transform_plane_to_local(gt[i], add_plane_noise(planes_gt[0], 0)), 0, i])
        # measurement2 = np.hstack([transform_plane_to_local(gt[i], add_plane_noise(planes_gt[1], 0)), 1, i])
        # measurement3 = np.hstack([transform_plane_to_local(gt[i], add_plane_noise(planes_gt[2], 0)), 2, i])

        # landmark_measurements = np.vstack([measurement1, measurement2, measurement3])

        obj.add_landmark_measurement(landmark_measurements)
        obj.solve()

    print("OBJ SL: ", obj.s_l)
    print("GT: ", plane1, plane2)

    show_trajectory_and_planes(x, obj.s_x, gt,
                               obj.s_l, planes_gt, np.array([1, 3, 5]))
    # plt.figure()
    # plt.plot(x[:,0], x[:,1], label= "Odometry")
    # plt.plot(obj.s_x[:,0], obj.s_x[:,1], label="Corrected")
    # plt.plot(gt[:,0], gt[:,1], label="Ground Truth")
    # plt.show()


