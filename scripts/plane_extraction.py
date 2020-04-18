#Usage: python3 velodyne_dataloader.py path_to_velodyne_sync
import sys
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import copy
import open3d as o3d

class PlaneExtracter():
    def __init__(self):
        self.min_dist = 0.2
        self.min_number_of_points = 1000
        self.num_iter = 1000

    def extract_planes(self,cloud):
        cloud_open3d= o3d.geometry.PointCloud()
        cloud_open3d.points = o3d.utility.Vector3dVector(cloud[:,:3])
        cloud_open3d.paint_uniform_color([1,0,0])
        cloud_numpy = cloud[:,:3]
        pointcloud_list = []
        number_of_planes = 4
        plane_pointcloud_all = np.zeros((0,3))   
        for i in range(number_of_planes):
            plane = cloud_open3d.segment_plane(self.min_dist,self.min_number_of_points,self.num_iter)
            plane_pointcloud = o3d.geometry.PointCloud()
            plane_pointcloud.points = o3d.utility.Vector3dVector(cloud_numpy[plane[1],:3])
            plane_pointcloud.paint_uniform_color(np.random.rand(3))
            if np.abs(np.dot(plane[0][:3],np.array([0,1,0])))>0.2:
                #check if ground plane by checking dot product with [0,1,0] and exlude it
                plane_pointcloud_all=np.vstack((plane_pointcloud_all,np.asarray(plane_pointcloud.points)))
            cloud_numpy = cloud_numpy[~np.isin(np.arange(cloud_numpy.shape[0]),plane[1]),:3]
            cloud_open3d.points = o3d.utility.Vector3dVector(cloud_numpy)
    
        return plane_pointcloud_all
        # o3d.visualization.draw_geometries(pointcloud_list)

    def visualize_poindclouds(self,pcd1,pcd2):
        pcd1.paint_uniform_color([1,0,0])
        pcd2.paint_uniform_color([0,1,0])
        all_points = [pcd2,pcd1]
        o3d.visualization.draw_geometries(all_points)

def convert(x_s, y_s, z_s):
    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

def load_velodyne_timestep(f_bin):
    hits = []

    while True:    
        x_str = f_bin.read(2)
        if x_str == b'': # eof
            break

        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert(x, y, z)

        s = "%5.3f, %5.3f, %5.3f, %d, %d" % (x, y, z, i, l)
        hits += [[x, y, z,l]]

    f_bin.close()

    hits = np.asarray(hits)
    return hits

def main(args):
    #good frame 1335705713193323
    feature_extractor = PlaneExtracter()
    file_name_list = []
    for f in os.listdir(sys.argv[1]):
        file_name_list.append(int(f[:-4]))
    sorted_path_list = sorted(file_name_list)

    if len(sys.argv) is 3:
        start_at_desired_frame = True
        desired_time_stamp = int(sys.argv[2])
        print("starting at desired time stamp\n")
    else: start_at_desired_frame = False
    counter = 0
    planes_all = []
    transformation = np.eye(4)
    plane_extractor = PlaneExtracter()
    first_bin = open(sys.argv[1]+str(desired_time_stamp)+".bin", "rb")
    first_cloud = cloud = load_velodyne_timestep(first_bin)
    first_cloud_open3d= o3d.geometry.PointCloud()
    first_cloud_open3d.points = o3d.utility.Vector3dVector(first_cloud[:,:3])
    first_cloud_plane = plane_extractor.extract_planes(first_cloud)
    first_cloud_open3d.points = o3d.utility.Vector3dVector(first_cloud_plane[:,:3])

    for i,f in enumerate(sorted_path_list):
        if start_at_desired_frame is True and f-desired_time_stamp < 0:
            continue
        start_at_desired_frame = False
        f_bin = open(sys.argv[1]+str(f)+".bin", "rb")
        cloud = load_velodyne_timestep(f_bin)
        planes_all.append(plane_extractor.extract_planes(cloud))
        if len(planes_all) == 2:
            cloud_open3d_1= o3d.geometry.PointCloud()
            cloud_open3d_1.points = o3d.utility.Vector3dVector(planes_all[0])
            cloud_open3d_2= o3d.geometry.PointCloud()
            cloud_open3d_2.points = o3d.utility.Vector3dVector(planes_all[1])
            reg_p2l = o3d.registration.registration_icp(cloud_open3d_1, cloud_open3d_2, 0.02, np.eye(4),o3d.registration.TransformationEstimationPointToPoint())
            transformation = transformation @np.linalg.inv( np.array(reg_p2l.transformation))
            del planes_all[0]
            if f-desired_time_stamp > 1*10**6:
                print(transformation)
                transformed_point_cloud = copy.deepcopy(first_cloud_open3d)
                transformed_point_cloud.transform(transformation)
                
                artificial_transform = np.eye(4)
                artificial_transform[:3,-1] = np.array([10,10,1])
                transformed_point_cloud2 = copy.deepcopy(first_cloud_open3d)
                transformed_point_cloud2.transform(artificial_transform)

                plane_extractor.visualize_poindclouds(transformed_point_cloud,cloud_open3d_2)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
