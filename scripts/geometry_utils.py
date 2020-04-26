import numpy as np
import math

def rot2D(theta):
    return np.array([math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)]).reshape(2,2)

def rot3D(theta):
    o = np.eye(3)
    o[:2,:2] = rot2D(theta)
    return o

# TODO: pass in normalized n
def transform_plane_to_world(pose_w, planes):
    x,y,theta = pose_w
    h = np.array([x,y,1]).reshape(-1,1)
    n_w = np.dot(planes[:,:3], rot3D(theta).T)

    d_w = (-np.dot(n_w, h) + planes[:,3].reshape(-1,1))
    return np.hstack([n_w,d_w])

def computeH(pos):
    x, y, t = pos
    H = np.eye(3)
    H[:2,:2] = rot2D(t)
    H[0,-1] = x
    H[1,-1] = y
    return H

def transform_plane_to_local(pose_w, plane_w):
    n_w = plane_w[:3].reshape(-1,1)
    d_w = plane_w[-1]
    x,y,theta = pose_w
    h = np.array([x,y,1]).reshape(-1,1)
    n_l = np.dot(rot3D(theta), n_w).reshape(-1,)
    d_l = (np.dot(n_w.T, h) + d_w).reshape(-1,) / np.linalg.norm(n_w)
    return np.hstack([n_l,d_l])

def normalize_planes(planes):
    return planes[:, :4] / np.linalg.norm(planes[:,:3], axis=1).reshape(-1, 1)

def midpoint(pt1, pt2):
    return (pt2 + pt1) / 2

def length_line(pt1, pt2):
    return np.linalg.norm(pt2 - pt1)

def parametric_line(pt1, pt2):
    pt1_h = np.hstack([pt1, 1])
    pt2_h = np.hstack([pt2, 1])
    line = np.cross(pt1_h, pt2_h)
    return line / np.linalg.norm(line[:2])


def plane_from_line(line, nz=5):
    nx, ny, d = line
    return np.array([nx, ny, nz, d])
