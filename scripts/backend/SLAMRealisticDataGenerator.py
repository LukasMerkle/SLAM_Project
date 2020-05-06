import numpy as np
from SLAMBackend import SLAMBackend
from geometry_utils import transform_plane_to_world, transform_plane_to_local, rot3D, \
                           normalize_planes, computeH
import copy
from grapher import show_trajectory, show_planes, show_trajectory_and_planes
import math

def measurement_noise(std):
    return np.random.normal(0,std, 4)


def add_plane_noise(planes, std):
    normalized_planes = normalize_planes((planes + measurement_noise(std)).reshape(1, -1))
    return normalized_planes.reshape(-1,)

def compute_world_pose(pos1, pos2):
    H1 = computeH(pos1)
    H2 = computeH(pos2)
    H = np.dot(H1, H2)
    t = math.atan2(H[1,0], H[0,0])
    x = H[0,-1]
    y = H[1,-1]
    return np.array([x,y,t])

class SimulateData():
    def __init__(self, std_trans=0.3, std_rot=0.1, std_theta=0.01, std_d=0.005):
        self.std_trans = std_trans
        self.std_rot = std_rot
        self.std_theta = std_theta
        self.std_d = std_d
        self.std_l = np.array([std_theta, std_theta, std_theta, std_d])

    def simulate_random(self, world_poses_gt, planes_gt, move_window=5, planes_per_pose=5):
        num_poses = world_poses_gt.shape[0]
        odom_gt, odom_sim = self.simulate_odometry(world_poses_gt)
        landmark_measurements = []
        count = 0
        n = 0
        for i in range(num_poses - 1):
            noisy_planes = np.vstack([np.hstack([transform_plane_to_local(world_poses_gt[i],
                                                          add_plane_noise(plane, self.std_l)), i, j])
                                                          for j, plane in enumerate(planes_gt)])
            if count == move_window:
                n += 1
                count = 0
            landmark_measurements.append(noisy_planes[n * planes_per_pose:(n + 1) * planes_per_pose,:])
            count += 1
        return odom_sim, np.vstack(landmark_measurements)

    # assumes p_idx then l_idx
    def get_noisy_planes(self, pose, plane, l_idx, p_idx):
        return np.hstack([transform_plane_to_local(pose, add_plane_noise(plane, self.std_l)), p_idx, l_idx])

    def simulate_odometry(self, gt_poses):
        num_poses = gt_poses.shape[0]
        odom_gt = np.vstack([SLAMBackend.odom_model(gt_poses[i], gt_poses[i + 1])
                             for i in range(gt_poses.shape[0] - 1)])
        odom_sim = odom_gt + np.hstack([np.random.normal(0, self.std_trans, (num_poses - 1, 2)),
                                        np.random.normal(0, self.std_rot, (num_poses - 1, 1))])
        return odom_gt, np.vstack([np.array([0, 0, 0]), odom_sim])

    # always sees closest_n planes
    # assumes odom_gt has a zero pose
    def simulate_based_on_closest(self, gt_poses, planes_gt, midpoints, closest_n):
        num_planes = planes_gt.shape[0]
        odom_gt, odom_sim = self.simulate_odometry(gt_poses)

        landmark_measurements = []
        plane_map = {i : -1 for i in range(num_planes)}
        seen = 0
        for p_idx, pose in enumerate(gt_poses):
            distances = np.linalg.norm(midpoints - pose[:2], axis=1)
            selected_planes = np.argpartition(distances, closest_n)[:closest_n]
            mapped_planes_idx = []
            for s in selected_planes:
                if plane_map[s] == -1:
                    plane_map[s] = seen
                    mapped_planes_idx.append(seen)
                    seen += 1
                else:
                    mapped_planes_idx.append(plane_map[s])
            planes_seen_in_frame = planes_gt[selected_planes]
            noisy_planes = np.vstack([self.get_noisy_planes(pose, plane, l_idx, p_idx)
                                      for plane, l_idx in zip(planes_seen_in_frame, mapped_planes_idx)])
            landmark_measurements.append(noisy_planes)
        return odom_sim, np.vstack(landmark_measurements)

if __name__ == "__main__":
    np.random.seed(0)

    num_poses = 15
    odom_list = np.zeros((num_poses, 3))
    odom_list[:, :2] = 1
    odom_list[:,-1] = np.deg2rad(10)
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
        obj.add_landmark_measurement(landmark_measurements[n*5:(n+1)*5,:])
        count +=1
        # import pdb; pdb.set_trace()
        obj.solve()

    show_trajectory(x, obj.s_x, gt)
    #obj.s_l,  planes_gt[:15,:], np.ones(planes_gt[:15,:].shape[0]))

    # plt.figure()
    # plt.plot(x[:,0], x[:,1], label= "Odometry")
    # plt.plot(obj.s_x[:,0], obj.s_x[:,1], label="Corrected")
    # plt.plot(obj.s_x[10,0], obj.s_x[10,1], 'ro')
    # plt.plot(gt[:,0], gt[:,1], label="Ground Truth")
    # plt.legend()
    # plt.show()


