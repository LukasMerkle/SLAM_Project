import sys
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import copy
import open3d as o3d
from plane_extraction import load_velodyne_timestep

f_bin = open(sys.argv[1]+"1335705713193323.bin", "rb")
cloud1 = load_velodyne_timestep(f_bin)
cloud_open3d_1= o3d.geometry.PointCloud()
cloud_open3d_1.points = o3d.utility.Vector3dVector(cloud1[:,:3])

f_bin2 = open(sys.argv[1]+"1335705714193377.bin", "rb")
cloud2 = load_velodyne_timestep(f_bin2)
cloud_open3d_2= o3d.geometry.PointCloud()
cloud_open3d_2.points = o3d.utility.Vector3dVector(cloud2[:,:3])

reg_p2l = o3d.registration.registration_icp(cloud_open3d_1, cloud_open3d_2, 5, np.eye(4),o3d.registration.TransformationEstimationPointToPoint(),o3d.registration.ICPConvergenceCriteria(max_iteration = 2000))
evaluation = o3d.registration.evaluate_registration(cloud_open3d_1, cloud_open3d_2, 0.02, reg_p2l.transformation)
print(evaluation)

transformed_point_cloud = copy.deepcopy(cloud_open3d_1)
transformation = reg_p2l.transformation
transformed_point_cloud.transform(transformation)

transformed_point_cloud.paint_uniform_color([1,0,0])
cloud_open3d_2.paint_uniform_color([0,1,0])
all_points = [cloud_open3d_2,transformed_point_cloud]
o3d.visualization.draw_geometries(all_points)
