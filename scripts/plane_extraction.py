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
        plane_normals = np.zeros((0,4))   
        for i in range(number_of_planes):
            plane = cloud_open3d.segment_plane(self.min_dist,self.min_number_of_points,self.num_iter)
            plane_pointcloud = o3d.geometry.PointCloud()
            plane_pointcloud.points = o3d.utility.Vector3dVector(cloud_numpy[plane[1],:3])
            plane_pointcloud.paint_uniform_color(np.random.rand(3))
            if np.abs(np.dot(plane[0][:3],np.array([0,1,0])))>0.2:
                #check if ground plane by checking dot product with [0,1,0] and exlude it
                plane_normals=np.vstack((plane_normals,plane[0]))
            cloud_numpy = cloud_numpy[~np.isin(np.arange(cloud_numpy.shape[0]),plane[1]),:3]
            cloud_open3d.points = o3d.utility.Vector3dVector(cloud_numpy)
    
        return plane_normals
        # o3d.visualization.draw_geometries(pointcloud_list)

def visualize_poindclouds(pcd1,pcd2):
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

def getPlaneCorrespondences(plane_dict,cloud_source,cloud_target,transformation):
    plane_extractor = PlaneExtracter()
    source_cloud_planes = plane_extractor.extract_planes(cloud_source)
    target_cloud_planes = plane_extractor.extract_planes(cloud_target)
    transformed_cloud_planes = transformation @ source_cloud_planes 
    for normals_source in transformed_cloud_planes:
        for normals_target in target_cloud_planes:
            if (np.linalg.norm(np.cross(normals_source[:3],normals_target[:3])) < 0.001 and np.abs(normals_source[3] - normals_target[3])):
                print("Same")






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

def sort_paths(directory):
    #sorts files with increasing timestamps
    file_name_list = []
    for f in os.listdir(directory):
        file_name_list.append(int(f[:-4]))
    sorted_path_list = sorted(file_name_list)
    return sorted_path_list

def create_open3d_pointcloud(cloud):
    #creates open3d pointcloud from numpy pointcloud
    cloud_open3d= o3d.geometry.PointCloud()
    cloud_open3d.points = o3d.utility.Vector3dVector(cloud)
    return cloud_open3d

def main(args):
    #good frame 1335705713193323

    if len(sys.argv) is 3:
        start_at_desired_frame = True
        desired_time_stamp = int(sys.argv[2])
        print("starting at desired time stamp\n")
    else: start_at_desired_frame = False
    
    planes_all = []
    transformation = np.eye(4)
    plane_extractor = PlaneExtracter()
    
    #Get start frame for ICP evaluation
    first_bin = open(sys.argv[1]+str(desired_time_stamp)+".bin", "rb")
    first_cloud = load_velodyne_timestep(first_bin)
    first_cloud_open3d= o3d.geometry.PointCloud()
    first_cloud_open3d.points = o3d.utility.Vector3dVector(first_cloud[:,:3])
    recent_clouds = []

    #Loop through all lidar scans and start with desired scan
    for i,f in enumerate(sort_paths(sys.argv[1])):
        #check if current scan is the desired start scan
        if start_at_desired_frame is True and f-desired_time_stamp < 0:
            continue
        start_at_desired_frame = False
        f_bin = open(sys.argv[1]+str(f)+".bin", "rb")
        cloud = load_velodyne_timestep(f_bin)
        recent_clouds.append(cloud[:,:3])

        #Skip ICP if for the first frame
        if len(recent_clouds) is not 2:
            continue

        cloud_open3d_1=create_open3d_pointcloud(recent_clouds[0])
        cloud_open3d_2=create_open3d_pointcloud(recent_clouds[1])

        reg_icp = o3d.registration.registration_icp(cloud_open3d_1, cloud_open3d_2, 2, np.eye(4),o3d.registration.TransformationEstimationPointToPoint(),o3d.registration.ICPConvergenceCriteria(max_iteration = 200))
        transformation = np.array(reg_icp.transformation) @ transformation
        del recent_clouds[0]
        
        #TEST AFTER N SECONDS
        if f-desired_time_stamp >  2*10**6:
            print(np.linalg.inv(transformation))
            transformed_point_cloud = copy.deepcopy(first_cloud_open3d)
            transformed_point_cloud.transform(transformation)
            visualize_poindclouds(transformed_point_cloud,cloud_open3d_2)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
