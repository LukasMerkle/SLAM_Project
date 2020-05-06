import numpy as np
import scipy.linalg

def gauss_newton(ABfunction, s_x, s_l, odom, landmark_measurements, std_x, std_l, max_iter=1000):
    for i in range(max_iter):
        A,b = ABfunction(s_x, s_l, odom, landmark_measurements, std_x, std_l)
        if(i == 1):
            print("START ERROR: ", np.linalg.norm(b))
        dx = scipy.linalg.solve(np.dot(A.T,A),np.dot(A.T,b))
        s_x_new = s_x + dx[:(odom.shape[0] + 1) * 3].reshape(-1, 3) # dim_x
        s_l_new = s_l + dx[(odom.shape[0] + 1) * 3:].reshape(-1, 4) # dim_l
        _,b_new = ABfunction(s_x_new, s_l_new, odom, landmark_measurements, std_x, std_l)
        if(np.linalg.norm(b_new) > np.linalg.norm(b)):
            print("Error going up - breaking", i)
            return s_x, s_l, np.linalg.norm(b)
        s_x = s_x_new
        s_l = s_l_new
        if(np.linalg.norm(b_new - b) < 1e-12):
            print("Converged",i)
            break
    return s_x, s_l, np.linalg.norm(b_new)

def lm(ABfunction, s, odom1, landmark_measurements, std_x, std_l, max_iter=1000):
    lambda_lm = 10

    for i in range(max_iter):
        s_x1 = s[:6]
        s_l = s[6:]
        A,b = ABfunction(s_x1, s_l, odom1, landmark_measurements, std_x, std_l)
        dx = scipy.linalg.solve(np.dot(A.T,A) + np.eye(A.shape[1]) * lambda_lm, np.dot(A.T,b))
        s_new = s + dx
        _,b_new = ABfunction(s_new[:6], s_new[6:], odom1, landmark_measurements, std_x, std_l)
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