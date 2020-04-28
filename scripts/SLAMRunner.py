import numpy as np
from SLAMBackend import SLAMBackend
from geometry_utils import transform_plane_to_world, transform_plane_to_local, rot3D, \
                           normalize_planes, computeH, compute_world_pose
from grapher import show_trajectory, show_trajectory_and_planes
import matplotlib.pyplot as plt

# TODO: allow for adding odometry and measurements gradually!!!!
class SlamRunner():
    def __init__(self, std_x, std_l, std_p, init_pose):
        self.std_x = std_x
        self.std_l = std_l
        self.std_p = std_p
        self.init_pose = init_pose
        self.curr_pose = 0
        self.gt_pose = None
        self.odometry = None
        self.total_odometry = None
        self.plane_gt = None
        self.plane_measurements = None
        self.obj = SLAMBackend(std_p, std_x, std_l, init_pose)

    def add_odometry(self, odometry, gt_pose):
        total_odometry = [self.init_pose]
        for i, odom in enumerate(odometry):
            total_odometry.append(compute_world_pose(total_odometry[-1], odom))
        self.total_odometry = np.vstack([total_odometry])
        self.gt_pose = gt_pose
        self.odometry = odometry

    def add_plane_measurements(self, plane_measurements, plane_gt=None):
        if(plane_gt is not None):
            self.plane_gt = plane_gt
        self.plane_measurements = plane_measurements

    def run(self, num_iter_to_break=None):
        for i,odom in enumerate(self.odometry):
            if (i % 10 == 0):
                print("Iteration:", i)
            if (num_iter_to_break is not None and i == num_iter_to_break):
                break
            self.obj.add_pose_measurement(odom)

            # see all planes always
            current_plane_inds = np.where(self.plane_measurements[:, self.obj.P_IDX] == i)[0]
            landmark_measurements = self.plane_measurements[current_plane_inds,:]

            self.obj.add_landmark_measurement(landmark_measurements)
            self.obj.solve()

    def save(self):
        np.save('corrected_sx', self.obj.s_x)
        np.save('corrected_sl', self.obj.s_l)

    def show(self, max_pose_to_graph=-1, show_planes=False, midpoints=None, size=None):
        if(max_pose_to_graph == -1):
            max_pose_to_graph = self.total_odometry.shape[0]
        if(show_planes is False or self.plane_gt is None):
            show_trajectory(self.total_odometry[:max_pose_to_graph],
                            self.obj.s_x[:max_pose_to_graph],
                            self.gt_pose[:max_pose_to_graph])
        else:
            if(midpoints is None):
                # for now just put them in a random spot
                midpoints = np.ones((self.plane_gt.shape[0], 2))
            if(size  is None):
                size = 1
            show_trajectory_and_planes(self.total_odometry, self.obj.s_x, self.gt_pose,
                                       self.plane_gt, self.plane_gt, midpoints, size)


if __name__ == "__main__":
    std_x = np.array([0.05, 0.05, 1e-3]) # x, y, theta
    std_l = np.array([1e-5, 1e-5, 1e-5, 0.05]) # nx, ny, nz, d
    std_p = np.array([0.05, 0.05, 0.001])
    init_pose = np.array([0,0,0])

    dic = np.load('planes_and_poses.npz')
    all_planes = dic['planes']
    odom_list = dic['poses']

    gt_dic = np.load('ground_truth.npz')
    gt = gt_dic['gt_trajectory']
    obj = SLAMBackend(std_p, std_x, std_l, init_pose)


    pure_odometry = [init_pose]
    for i,odom in enumerate(odom_list):
        pure_odometry.append(compute_world_pose(pure_odometry[-1], odom))
    pure_odometry = np.vstack([pure_odometry])

    # np.save('pure_odometry', pure_odometry)

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
    plt.gca().set_aspect('equal')
    plt.plot(pure_odometry[:,0], pure_odometry[:,1], label= "Odometry")
    plt.plot(s_x[:,0], s_x[:,1], label="Corrected")
    plt.plot(gt[:500,0], gt[:500,1], label="Ground Truth")
    plt.legend()
    plt.show()


