# from optimizers import lm, gaussnewton
import numpy as np
from geometry_utils import transform_plane_to_world, transform_plane_to_local, rot3D
import copy
import scipy.linalg

class SLAMBackend:
    L_IDX = 4
    P_IDX = 5
    dim_x = 3
    dim_l = 4
    def __init__(self, std_p, std_x, std_l, init_pose):
        self.s_x = init_pose.reshape(1,-1) # num_poses x 3
        self.s_l = None # num_landmarks x 4
        self.odom = [] # num_odom_measurements x 3
        self.landmarks = [] # num_landmark_measurements x 4
        self.std_p = std_p
        self.std_x = std_x
        self.std_l = std_l

    def landmark_model(self, pose_w, planes):
        return transform_plane_to_local(pose_w, planes)

    def odom_model(self, pos1, pos2):
        return pos2-pos1

    def odom_jacobian(self, pos1, pos2):
        x1, y1, t1 = pos1
        x2, y2, t2 = pos2
        H = np.zeros((3,6))
        H[0,0] = -1
        H[0,3] = 1
        H[1,1] = -1
        H[1,4] = 1
        H[2,2] = -1
        H[2,5] = 1
        return H

    def numeraical_jacobian(self, pos1, pos2, model):
        eps = 1e-12
        pos_H = []
        neg_H = []
        for i,x in enumerate(pos1):
            v = copy.copy(pos1)
            v[i] = x + eps
            pos_H.append(model(v, pos2))
        for i,x in enumerate(pos2):
            v = copy.copy(pos2)
            v[i] = x + eps
            pos_H.append(model(pos1, v))

        for i,x in enumerate(pos1):
            v = copy.copy(pos1)
            v[i] = x - eps
            neg_H.append(model(v, pos2))
        for i,x in enumerate(pos2):
            v = copy.copy(pos2)
            v[i] = x - eps
            neg_H.append(model(pos1, v))

        H = (np.vstack(pos_H).T - np.vstack(neg_H).T) / (2*eps)
        return H

    def generateAB(self, s_x, s_l):
        num_poses = s_x.shape[0]
        num_l = s_l.shape[0]
        num_l_measurements = len(self.landmarks)

        A_prior = np.zeros((3, s_x.shape[0]*self.dim_x + s_l.shape[0]*self.dim_l))
        A_prior[:,:3] = np.eye(3) / np.sqrt(self.std_p)

        A_odom = [A_prior]
        p_odom = [s_x[0]]
        for i,o in enumerate(self.odom):
            odom_jac = self.odom_jacobian(s_x[i], s_x[i + 1]) / np.sqrt(self.std_x).reshape(-1,1)
            A_sub = np.hstack([np.zeros((self.dim_x,i*self.dim_x)), odom_jac,
                            np.zeros((self.dim_x,(num_poses-i-2)*self.dim_x)),
                            np.zeros((self.dim_x, num_l*self.dim_l))])
            A_odom.append(A_sub)
            p_odom.append(self.odom_model(s_x[i], s_x[i + 1]))
        A_odom = np.vstack(A_odom)
        p_odom = np.hstack(p_odom).reshape(-1,1)
        m_odom = np.vstack([np.array([0,0,0]), self.odom]).reshape(-1,1)
        std_x_repeated = np.tile(self.std_x, num_poses - 1)
        b_odom = (m_odom - p_odom) / np.sqrt(np.hstack([self.std_p, std_x_repeated]).reshape(-1, 1))

        A_landmark = []
        p_landmark = []
        for i,l in enumerate(self.landmarks):
            # import pdb; pdb.set_trace()
            l_id = int(l[self.L_IDX])
            p_id = int(l[self.P_IDX])
            measurement_jac = self.numeraical_jacobian(s_x[p_id], s_l[l_id], self.landmark_model) / np.sqrt(self.std_l.reshape(-1, 1))
            A_sub = np.hstack([np.zeros((self.dim_l,p_id*self.dim_x)), measurement_jac[:,:3],
                            np.zeros((self.dim_l,(num_poses-p_id-1)*self.dim_x)),
                            np.zeros((self.dim_l,l_id*self.dim_l)), measurement_jac[:,3:],np.zeros((self.dim_l,(num_l-l_id-1)*self.dim_l))])
            A_landmark.append(A_sub)
            p_landmark.append(self.landmark_model(s_x[p_id], s_l[l_id]))
        A_landmark = np.vstack(A_landmark)
        p_landmark = np.vstack(p_landmark).reshape(-1,1)

        m_landmark = np.vstack(self.landmarks)[:, :-2].reshape(-1, 1)
        std_l_repeated = np.tile(self.std_l, num_l_measurements)
        b_landmark = (m_landmark - p_landmark) / np.sqrt(std_l_repeated.reshape(-1, 1))
        A = np.vstack([A_odom, A_landmark])
        b = np.vstack([b_odom, b_landmark]).reshape(-1,)
        return A,b

    def gauss_newton(self, max_iter=1000):
        s_x = copy.copy(self.s_x)
        s_l = copy.copy(self.s_l)

        for i in range(max_iter):
            A,b = self.generateAB(s_x, s_l)
            if(i == 1):
                print("START ERROR: ", np.linalg.norm(b))
            dx = scipy.linalg.solve(np.dot(A.T,A),np.dot(A.T,b))
            s_x_new = s_x + dx[:(len(self.odom) + 1) * 3].reshape(-1, self.dim_x) # dim_x
            s_l_new = s_l + dx[(len(self.odom) + 1) * 3:].reshape(-1, self.dim_l) # dim_l
            _,b_new = self.generateAB(s_x_new, s_l_new)
            if(np.linalg.norm(b_new) > np.linalg.norm(b)):
                print("Error going up - breaking", i)
                return s_x, s_l, np.linalg.norm(b)
            s_x = s_x_new
            s_l = s_l_new
            if(np.linalg.norm(b_new - b) < 1e-12):
                print("Converged",i)
                break
        return s_x, s_l, np.linalg.norm(b_new)

    #-> appends to odom_list, initalize new pose in s_x
    def add_pose_measurement(self, odom_measurement):
        self.s_x = np.vstack([self.s_x, self.s_x[-1] + odom_measurement])
        self.odom.append(odom_measurement)

    #-> appends to landmark_measurements, initializes new landmark in self.s_l
    # assumes you get odometry first
    # assumes landmarks have normalized n's because of prediction - measurement calculation
    def add_landmark_measurement(self, landmark_measurement):
        if self.s_l is not None:
            new_indices = landmark_measurement[np.where(landmark_measurement[:,self.L_IDX] >= self.s_l.shape[0])[0], self.L_IDX]
            if new_indices.shape[0] > 0:
                new_measurements = landmark_measurement[np.sort(new_indices),:] #sorting new indices by index
                self.s_l = np.vstack([self.s_l, transform_plane_to_world(self.s_x[-1],new_measurements)])
        else:
            self.s_l = transform_plane_to_world(self.s_x[-1],landmark_measurement)

        for x in landmark_measurement:
            self.landmarks.append(x)

    def solve(self):
        self.s_x, self.s_l, _ = self.gauss_newton()

if __name__ == "__main__":
    std_x = np.array([1, 1, .1]) # x, y, theta
    std_l = np.array([0.01, 0.01, 0.01, 0.01]) # nx, ny, nz, d
    std_p = np.array([0.05, 0.05, 0.001])
    init_pose = np.array([0,0,0])

    obj = SLAMBackend(std_p, std_x, std_l, init_pose)
    print(obj.s_x)
    print(obj.s_l)
    print(obj.odom)

    x1 = np.array([2,3,np.deg2rad(45)])
    odom1 = x1 - init_pose

    plane1 = np.array([0.707, 0.707, 0, 5]) / np.linalg.norm([0.707, 0.707])
    plane2 = np.array([-0.707, 0.707, 0, 2]) / np.linalg.norm([0.707, 0.707])
    measurement11 = np.hstack([transform_plane_to_local(init_pose, plane1), 0, 0])
    measurement12 = np.hstack([transform_plane_to_local(init_pose, plane2), 1, 0])
    measurement21 = np.hstack([transform_plane_to_local(x1, plane1), 0, 1])
    measurement22 = np.hstack([transform_plane_to_local(x1, plane2), 1, 1])
    landmark_measurements = np.vstack([measurement11, measurement12, measurement21, measurement22])

    obj.add_landmark_measurement(landmark_measurements[:2, :])

    print(obj.s_l)
    print(obj.landmarks)
    obj.add_pose_measurement(odom1)

    obj.add_landmark_measurement(landmark_measurements[2:, :])
    print(obj.s_x)
    print(obj.s_l)
    print(obj.landmarks)
    A, b = obj.generateAB(obj.s_x, obj.s_l)
    #print(b)
    import pdb; pdb.set_trace()
    #obj.solve()
    print(obj.s_x)
    print(obj.s_l)

