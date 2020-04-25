import numpy as np
import matplotlib.pyplot as plt
from SLAMBackend import SLAMBackend
from geometry_utils import transform_plane_to_world, transform_plane_to_local, rot3D
import copy

def measurement_noise(std):
    return np.random.normal(0,std, 4)

if __name__ == "__main__":
    num_poses = 100
    odom_list = np.zeros((num_poses, 3))
    odom_list[:, :2] = 1

    gt_odom = copy.deepcopy(odom_list)
    gt = np.cumsum(gt_odom, axis=0)

    odom_list += np.hstack([np.random.normal(0, 1, (num_poses, 2)), np.random.normal(0, 0.1, (num_poses, 1))])

    x = np.cumsum(odom_list, axis=0)

    std_x = np.array([1, 1, .1]) # x, y, theta
    std_l = np.array([0.01, 0.01, 0.01, 0.01]) # nx, ny, nz, d
    std_p = np.array([0.05, 0.05, 0.001])
    init_pose = np.array([0,0,0])

    plane1 = np.array([0.707, 0.707, 0, 5])
    plane2 = np.array([-0.707, 0.707, 0, 2])

    obj = SLAMBackend(std_p, std_x, std_l, init_pose)

    for i,odom in enumerate(odom_list):
        print("Iteration:", i)
        obj.add_pose_measurement(odom)

        measurement1 = np.hstack([transform_plane_to_local(gt[i], plane1), 0, i])
        measurement2 = np.hstack([transform_plane_to_local(gt[i], plane2), 1, i])

        landmark_measurements = np.vstack([measurement1, measurement2])

        obj.add_landmark_measurement(landmark_measurements)

        obj.solve()

    plt.figure()
    plt.plot(x[:,0], x[:,1], label= "Odometry")
    plt.plot(obj.s_x[:,0], obj.s_x[:,1], label="Corrected")
    plt.plot(gt[:,0], gt[:,1], label="Ground Truth")
    plt.show()


