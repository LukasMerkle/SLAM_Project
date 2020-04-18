import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import scipy.linalg
import pandas as pd

# TODO:
# programatically create A and b (pipeline)
# visualizations to show convergence or not (2d planes + our pose vs. gt)


L_IDX = 4
P_IDX = 5

def main():
    x0 = np.array([0,0,0])
    x1 = np.array([2,3,np.deg2rad(45)])
    x2 = np.array([5,5,np.deg2rad(-45)])

    plane1 = np.array([0.707, 0.707, 0, 5])
    plane2 = np.array([-0.707, 0.707, 0, 2])

    gt = np.hstack([x0, x1, plane1, plane2])

    odom1 = odom_model(x0,x1) + odom_noise()
    odom2 = odom_model(x1,x2) + odom_noise()

    # MEASUREMENT: [nx, ny, nz, d, id_landmark, id_pose]
    measurement11 = np.hstack([measurement_model_w(x0, plane1) + measurement_noise(0), 1, 1])
    measurement12 = np.hstack([measurement_model_w(x0, plane2) + measurement_noise(0), 2, 1])
    measurement21 = np.hstack([measurement_model_w(x1, plane1) + measurement_noise(0), 1, 2])
    measurement22 = np.hstack([measurement_model_w(x1, plane2) + measurement_noise(0), 2, 2])
    landmark_measurements = np.hstack([measurement11, measurement12, measurement21, measurement22])

    n1 = plane1[:3] + np.random.normal(0,0.1, 3)
    n1 /= np.linalg.norm(n1)
    plane1[:3] = n1

    n2 = plane2[:3] + np.random.normal(0,0.1, 3)
    n2 /= np.linalg.norm(n2)
    plane2[:3] = n2

    init_measurement1 = measurement_model_w(x0, plane1)
    init_measurement2 = measurement_model_w(x0, plane2)

    #std_x = np.array([0.5, 0.5, 0.1]) # x, y, theta
    std_x = np.array([0.6, 0.6, .1]) # x, y, theta
    #std_l = np.array([1, 1, 1, 1]) # nx, ny, nz, d
    std_l = np.array([0.1, 0.1, 0.1, 0.1]) # nx, ny, nz, d

    s_x0 = x0
    s_l = np.hstack([invert_measurement_l(x0, init_measurement1), invert_measurement_l(x0,init_measurement2)])
    print(s_l)
    s_x1 = np.hstack([x0, x0+odom1])

    s = np.hstack([s_x1, s_l])
    print(s)

    A_hard,b_hard = hardcode_generateAB(s_x1, s_l, odom1, landmark_measurements, std_x, std_l)
    print(A.shape, b.shape, s.shape)


    out_g, err_g = gauss_newton(s, odom1, landmark_measurements, std_x, std_l)
    out_lm, err_lm = lm(s, odom1, landmark_measurements, std_x, std_l)
    print(np.round(s, 2))
    print(np.round(out_g,2))
    print(np.round(out_lm,2))
    print(gt)

    print("ACTUAL ERRORS: ")
    print(err_g)
    print(err_lm)

    s_error = np.linalg.norm(gt-s)
    og_error = np.linalg.norm(gt-out_g)
    olm_error = np.linalg.norm(gt-out_lm)
    print(s_error, og_error, olm_error)

    print(np.round(abs(gt[:6]-out_g[:6]),3))
    print(np.round(abs(gt[:6]-out_lm[:6]),3))

    x_results_pd = pd.DataFrame(np.vstack([s[:6], out_g[:6], out_lm[:6], gt[:6]]))
    l_results_pd = pd.DataFrame(np.vstack([s[6:], out_g[6:], out_lm[6:], gt[6:]]))
    print("STATE!!!!")
    print(x_results_pd)

    print("LANDMARKS!!!!")
    print(l_results_pd)


def rot2D(theta):
    return np.array([math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)]).reshape(2,2)

def rot3D(theta):
    o = np.eye(3)
    o[:2,:2] = rot2D(theta)
    return o

def numeraical_jacobian(pos1, pos2, model):
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


def odom_model(pos1, pos2):
#     x1, y1, t1 = pos1
#     x2, y2, t2 = pos2
#     H = np.eye(3)
#     R = rot2D(t2-t1)
#     odom = pos2[:2] - np.dot(R, pos1[:2].reshape(-1,1)).reshape(-1,)
    return pos2-pos1

def odom_jacobian(pos1, pos2):
    x1, y1, t1 = pos1
    x2, y2, t2 = pos2
#     H = np.array([[-math.cos(t2-t1), math.sin(t2-t1), -x1*math.sin(t2-t1) - y1*math.cos(t2-t1),
#                   1, 0, +x1*math.sin(t2-t1) + y1*math.cos(t2-t1)],
#                  [-math.sin(t2-t1), -math.cos(t2-t1), x1*math.cos(t2-t1) - y1*math.sin(t2-t1),
#                   0, 1, -x1*math.cos(t2-t1) + y1*math.sin(t2-t1)],
#                  [0, 0, -1, 0, 0, 1]])
    H = np.zeros((3,6))
    H[0,0] = -1
    H[0,3] = 1
    H[1,1] = -1
    H[1,4] = 1
    H[2,2] = -1
    H[2,5] = 1
    return H

def measurement_model_w(pose_w, plane_w):
    n_w = plane_w[:3].reshape(-1,1)
    d_w = plane_w[-1]
    x,y,theta = pose_w
    h = np.array([x,y,1]).reshape(-1,1)
    n_l = np.dot(rot3D(theta), n_w).reshape(-1,)
    d_l = (np.dot(n_w.T, h) + d_w).reshape(-1,) / np.linalg.norm(n_w)
    return np.hstack([n_l,d_l])

# TODO: pass in normalized n
def invert_measurement_l(pose_w, plane_l):
    n_l = plane_l[:3].reshape(-1,1)
    d_l = plane_l[-1]
    x,y,theta = pose_w
    h = np.array([x,y,1]).reshape(-1,1)
    n_w = np.dot(rot3D(theta).T, n_l).reshape(-1,)

    d_w = (-np.dot(n_w.T, h) + d_l).reshape(-1,)
    return np.hstack([n_w,d_w])

def measurement_jacobian(pose, plane):
    x,y,theta = pose
    nx_w,ny_w,nz_w,d_w = plane

    H = np.array([[0, 0, -math.sin(theta)*nx_w - math.cos(theta)*ny_w, math.cos(theta), -math.sin(theta), 0, 0],
                  [0 ,0, math.cos(theta)*nx_w - math.sin(theta)*ny_w, math.sin(theta), math.cos(theta), 0, 0],
                  [0, 0, 0, 0, 0, 1, 0],
                  [nx_w, ny_w, 0, x, y, 1, 1]]) # re-compute normalized jacobian

    return H

def odom_noise():
    return np.hstack([np.random.normal(0, 0.6, 2), np.random.normal(0, 0.1, 1)])

def measurement_noise(std):
    return np.random.normal(0,std, 4)

# s_x: num_poses x dim_x
# s_l: num_l x dim_l
# landmark_measurements: num_l_meas x 6
# odom: num_odom x 3
def generateAB(s_x, s_l, odom, landmark_measurements, std_x, std_l, dim_x=3, dim_l=4):
    num_poses = s_x.shape[0]
    num_l = int(len(s_l) / dim_l)
    #num_l = len(np.unique(landmark_measurements[:, L_IDX]))
    num_odom = len(odom) + 1 # includes prior
    num_l_measurements = int(len(landmark_measurements) / dim_l)

    std_p = np.array([0.05, 0.05, 0.005]) # x, y, theta prior

    A_x = np.zeros((num_odom * dim_x, num_poses * dim_x + num_l * dim_l))
    A_x[:3,:3] = np.eye(3) / np.sqrt(std_p)

    p_x1 = s_x[0]
    # -1 because prior already accounted for
    for i in range(num_odom - 1):
        odom_jac = odom_jacobian(s_x[i], s_x[i + 1])
        std_x_repeated = np.tile(std_x, 2)
        A_x[3 * (i + 1):3 * (i + 2), 3 * i: 3 * (i + 2)] = odom_jac / np.sqrt(std_x_repeated)
        p_x1 = np.hstack([p_x1, odom_model(s_x[i], s_x[i + 1])])

    A_l1 = np.zeros((num_l_measurements * dim_l, num_poses * dim_x + num_l * dim_l))

    p_l1_list = []
    for i in range(num_l_measurements):
        l_id = landmark_measurements[i, L_IDX]
        p_id = landmark_measurements[i, P_IDX]
        measurement_jac = numeraical_jacobian(s_x[p_id], s_l[l_id], measurement_model_w)
        A_l1[4 * i:4 * (i + 1), p_id:p_id + 3] = measurement_jac[:, :3] / np.sqrt(std_l.reshape(-1, 1))
        A_l1[4 * i:4 * (i + 1), num_poses * 3 + l_id * 4:num_poses * 3 + (l_id + 1) * 4]
                                = measurement_jac[:, 3:7] / np.sqrt(std_l.reshape(-1, 1))

        p_l1_list.append(measurement_model_w(s_x[p_id], s_l[l_id]))

    p_l1 = np.vstack(p_l1_list)

    m_x1 = np.vstack([np.array([0,0,0]), odom]).reshape(-1, 1)
    std_x_repeated = np.tile(std_x, num_poses - 1)
    b_x1 = (m_x1 - p_x1) / np.sqrt(np.hstack([std_p, std_x_repeated]))

    m_l1 = landmark_measurements[:, :-2].reshape(-1, 1)
    std_l_repeated = np.tile(std_l, num_l_measurements)
    b_l1 = (m_l1 - p_l1) / np.sqrt(std_l_repeated)

    A = np.vstack([A_x, A_l1])
    b = np.hstack([b_x1, b_l1])
    return A,b


def hardcode_generateAB(s_x1, s_l, odom1, landmark_measurements, std_x, std_l, dim_x=3, dim_l=4):
    num_x = int(len(s_x1) / dim_x)
    num_l = int(len(s_l) / dim_l)
    num_l_measurements = int(len(landmark_measurements) / dim_l)

    std_p = np.array([0.05, 0.05, 0.005]) # x, y, theta prior
    A_x1 = np.zeros((6,14))
    A_x1[:3,:3] = np.eye(3) / np.sqrt(std_p)
    odom_jac = odom_jacobian(s_x1[0:3], s_x1[3:6])
    std_x_repeated = np.tile(std_x, 2)
    A_x1[3:,:6] = odom_jac / np.sqrt(std_x_repeated)

    A_l1 = np.zeros((16,14))
    #measurement_jac1 = measurement_jacobian(s_x1[:3], s_l[:4])
    measurement_jac1 = numeraical_jacobian(s_x1[:3], s_l[:4], measurement_model_w)
    A_l1[:4,0:3] = measurement_jac1[:, 0:3] / np.sqrt(std_x)
    A_l1[:4,6:10] = measurement_jac1[:, 3:7] / np.sqrt(std_l)

    #measurement_jac2 = measurement_jacobian(s_x1[:3], s_l[4:])
    measurement_jac2 = numeraical_jacobian(s_x1[:3], s_l[4:], measurement_model_w)
    A_l1[4:8,0:3] = measurement_jac2[:, 0:3] / np.sqrt(std_x)
    A_l1[4:8,10:14] = measurement_jac2[:, 3:7] / np.sqrt(std_l)

    #measurement_jac1 = measurement_jacobian(s_x1[3:], s_l[:4])
    measurement_jac1 = numeraical_jacobian(s_x1[3:], s_l[:4], measurement_model_w)
    A_l1[8:12,3:6] = measurement_jac1[:, 0:3] / np.sqrt(std_x)
    A_l1[8:12,6:10] = measurement_jac1[:, 3:7] / np.sqrt(std_l)
    #measurement_jac2 = measurement_jacobian(s_x1[3:], s_l[4:])
    measurement_jac2 = numeraical_jacobian(s_x1[3:], s_l[4:], measurement_model_w)
    A_l1[12:16,3:6] = measurement_jac2[:, 0:3] / np.sqrt(std_x)
    A_l1[12:16,10:14] = measurement_jac2[:, 3:7] / np.sqrt(std_l)

    m_x1 = np.hstack([np.array([0,0,0]), odom1])
    p_x1 = np.hstack([np.array([0,0,0]), odom_model(s_x1[:3], s_x1[3:])])
    std_x_repeated = np.tile(std_x, num_x - 1)
    b_x1 = (m_x1 - p_x1) / np.sqrt(np.hstack([std_p, std_x_repeated]))

    m_l1 = landmark_measurements
    p_l1 = np.hstack([measurement_model_w(s_x1[:3], s_l[:4]), measurement_model_w(s_x1[:3], s_l[4:8]),
                      measurement_model_w(s_x1[3:6], s_l[:4]), measurement_model_w(s_x1[3:6], s_l[4:8])])
    std_l_repeated = np.tile(std_l, num_l_measurements)
    b_l1 = (m_l1 - p_l1) / np.sqrt(std_l_repeated)

    A = np.vstack([A_x1, A_l1])
    b = np.hstack([b_x1, b_l1])
    return A,b

def gauss_newton(s, odom1, landmark_measurements, std_x, std_l, max_iter=1000):
    for i in range(max_iter):
        s_x1 = s[:6]
        s_l = s[6:]
        A,b = generateAB(s_x1, s_l, odom1, landmark_measurements, std_x, std_l)
        if(i == 1):
            print("START ERROR: ", np.linalg.norm(b))
        dx = scipy.linalg.solve(np.dot(A.T,A),np.dot(A.T,b))
        s_new = s + dx
        _,b_new = generateAB(s_new[:6], s_new[6:], odom1, landmark_measurements, std_x, std_l)
        if(np.linalg.norm(b_new) > np.linalg.norm(b)):
            print("Error going up - breaking", i)
            return s, np.linalg.norm(b)
        s = s_new
        if(np.linalg.norm(b_new - b) < 1e-12):
            print("Converged",i)
            break
    return s, np.linalg.norm(b_new)

def lm(s, odom1, landmark_measurements, std_x, std_l, max_iter=1000):
    lambda_lm = 10

    for i in range(max_iter):
        s_x1 = s[:6]
        s_l = s[6:]
        A,b = generateAB(s_x1, s_l, odom1, landmark_measurements, std_x, std_l)
        dx = scipy.linalg.solve(np.dot(A.T,A) + np.eye(A.shape[1]) * lambda_lm, np.dot(A.T,b))
        s_new = s + dx
        _,b_new = generateAB(s_new[:6], s_new[6:], odom1, landmark_measurements, std_x, std_l)
        if(np.linalg.norm(b_new) > np.linalg.norm(b)):
            lambda_lm *= 10
            continue
        else:
            lambda_lm /= 10
            s = s_new

        if(np.linalg.norm(b_new - b) < 1e-12):
            print("Converged",i)
            break
    return s, np.linalg.norm(b_new)

if __name__ == '__main__':
    main()
