# from optimizers import lm, gaussnewton
import numpy as np
from geometry_utils import transform_plane_to_world, transform_plane_to_local, rot3D

class SLAMBackend:
    L_IDX = 4
    P_IDX = 5
    def __init__(self, std_p, std_x, std_l, init_pose):
        self.s_x = init_pose.reshape(1,-1) # num_poses x 3
        self.s_l = None # num_landmarks x 4
        self.odom = [] # num_odom_measurements x 3
        self.landmarks = [] # num_landmark_measurements x 4
        self.std_p = std_p
        self.std_x = std_x
        self.std_l = std_l
        
    def predict_landmark(self, pose_w, planes):
        return transform_plane_to_local(pose_w, planes)

    #-> appends to odom_list, initalize new pose in s_x
    def add_pose_measurement(self, odom_measurement):
        self.s_x = np.vstack([self.s_x, self.s_x[-1] + odom_measurement])
        self.odom.append(odom_measurement)

    #-> appends to landmark_measurements, initializes new landmark in s_l
    # assumes you get odometry first
    # assumes landmarks have normalized n's
    def add_landmark_measurement(self, landmark_measurement):
        if self.s_l is not None:
            new_indices = landmark_measurement[np.where(landmark_measurement[:,self.L_IDX] >= self.s_l.shape[0])[0], self.L_IDX]
            if new_indices.shape[0] > 0:
                new_measurements = landmark_measurement[np.sort(new_indices),:] #sorting new indices by index
                self.s_l = np.vstack([self.s_l, transform_plane_to_world(self.s_x[-1],new_measurements)])
        else:
            self.s_l = transform_plane_to_world(self.s_x[-1],landmark_measurement)
        
        self.landmarks.append(landmark_measurement)

    def solve(self):
        raise NotImplementedError


if __name__ == "__main__":
    std_x = np.array([1, 1, .1]) # x, y, theta
    std_l = np.array([0.01, 0.01, 0.01, 0.01]) # nx, ny, nz, d
    std_p = np.array([0.05, 0.05, 0.001])
    init_pose = np.array([0,0,0])

    obj = SLAMBackend(std_p, std_x, std_l, init_pose)
    print(obj.s_x)
    print(obj.s_l)
    print(obj.odom)

    plane1 = np.array([0.707, 0.707, 0, 5])
    plane2 = np.array([-0.707, 0.707, 0, 2])
    measurement11 = np.hstack([transform_plane_to_local(init_pose, plane1) , 0, 0])
    measurement12 = np.hstack([transform_plane_to_local(init_pose, plane2), 1, 0])
    measurement21 = np.hstack([transform_plane_to_local(init_pose + np.array([5,4,3]), plane1) , 0, 1])
    measurement22 = np.hstack([transform_plane_to_local(init_pose + np.array([5,4,3]), plane2), 1, 1])
    landmark_measurements = np.vstack([measurement11, measurement12, measurement21, measurement22])

    obj.add_landmark_measurement(landmark_measurements[:2, :])

    print(obj.s_l)
    print(obj.landmarks)
    obj.add_pose_measurement(np.array([5,4,3]))

    obj.add_landmark_measurement(landmark_measurements[2:, :])
    print(obj.s_x)
    print(obj.s_l)
    print(obj.landmarks)



 
    