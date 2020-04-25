#Usage: python3 velodyne_dataloader.py path_to_velodyne_sync
import sys
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import copy
import open3d as o3d
import pickle
class PlaneExtracter():
    def __init__(self):
        self.min_dist = 0.2
        self.min_number_of_points = 1000
        self.num_iter = 1000

    def extract_planes(self,cloud):
        cloud_numpy = np.asarray(cloud.points)
        pointcloud_list = []
        number_of_planes = 4
        plane_normals = np.zeros((0,4))   
        for i in range(number_of_planes):
            plane = cloud.segment_plane(self.min_dist,self.min_number_of_points,self.num_iter)
            plane_pointcloud = o3d.geometry.PointCloud()
            plane_pointcloud.points = o3d.utility.Vector3dVector(cloud_numpy[plane[1],:3])
            plane_pointcloud.paint_uniform_color(np.random.rand(3))
            if np.abs(np.dot(plane[0][:3],np.array([0,1,0])))>0.2:
                #check if ground plane by checking dot product with [0,1,0] and exlude it
                plane_normals=np.vstack((plane_normals,plane[0]))
            cloud_numpy = cloud_numpy[~np.isin(np.arange(cloud_numpy.shape[0]),plane[1]),:3]
            cloud.points = o3d.utility.Vector3dVector(cloud_numpy)
    
        return plane_normals
        # o3d.visualization.draw_geometries(pointcloud_list)

    def getPlaneCorrespondences(self,cloud_source,cloud_target,transformation):
        plane_extractor = PlaneExtracter()
        source_cloud_planes = plane_extractor.extract_planes(cloud_source)
        target_cloud_planes = plane_extractor.extract_planes(cloud_target)
        transformed_cloud_planes = transformation @ np.hstack((source_cloud_planes[:,:3],np.ones((source_cloud_planes.shape[0],1)))).T 
        transformed_cloud_planes[-1,:] = source_cloud_planes[:,-1]
        transformed_cloud_planes = transformed_cloud_planes.T
        for normals_source in transformed_cloud_planes:
            for normals_target in target_cloud_planes:
                if (1-np.dot(normals_source[:3]/np.linalg.norm(normals_source[:3]),normals_target[:3])<0.1) and (np.abs(normals_source[3]-normals_target[3])<0.2):
                    print("same")
                    print("Source",normals_source)
                    print("Target",normals_target)


class DataLoader():
    def load_velodyne_timestep(self,f_bin):
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

            x, y, z = self.convert(x, y, z)

            s = "%5.3f, %5.3f, %5.3f, %d, %d" % (x, y, z, i, l)
            hits += [[x, y, z,l]]

        f_bin.close()

        hits = np.asarray(hits)
        return hits

    def convert(self,x_s, y_s, z_s):
        scaling = 0.005 # 5 mm
        offset = -100.0

        x = x_s * scaling + offset
        y = y_s * scaling + offset
        z = z_s * scaling + offset

        return x, y, z

    def sort_paths(self,directory):
        #sorts files with increasing timestamps
        file_name_list = []
        for f in os.listdir(directory):
            file_name_list.append(int(f[:-4]))
        sorted_path_list = sorted(file_name_list)
        return sorted_path_list

class PointCloudOperator():
    def visualize_poindclouds(self,pcd1,pcd2):
        pcd1.paint_uniform_color([1,0,0])
        pcd2.paint_uniform_color([0,1,0])
        all_points = [pcd2,pcd1]
        o3d.visualization.draw_geometries(all_points)

    def create_open3d_pointcloud(self,cloud):
        #creates open3d pointcloud from numpy pointcloud
        cloud_open3d= o3d.geometry.PointCloud()
        cloud_open3d.points = o3d.utility.Vector3dVector(cloud)
        cloud_open3d = cloud_open3d.uniform_down_sample(5)
        return cloud_open3d

    def evaluate_performance(self,transformation,first_cloud_open3d,cloud_open3d_2,trajectory):
        print(np.linalg.inv(transformation))
        with open('trajectory.pickle', 'wb') as f:
            pickle.dump(trajectory, f)
        transformed_point_cloud = copy.deepcopy(first_cloud_open3d)
        transformed_point_cloud.transform(transformation)
        self.visualize_poindclouds(transformed_point_cloud,cloud_open3d_2)


def ICP(cloud1,cloud2,point_cloud_operator):
    cloud_open3d_1=point_cloud_operator.create_open3d_pointcloud(cloud1)
    cloud_open3d_2=point_cloud_operator.create_open3d_pointcloud(cloud2)

    reg_icp = o3d.registration.registration_icp(cloud_open3d_1, cloud_open3d_2, 2, np.eye(4),o3d.registration.TransformationEstimationPointToPoint())    
    evaluation = o3d.registration.evaluate_registration(cloud_open3d_1, cloud_open3d_2, 2, np.eye(4))
    print(evaluation)
    return reg_icp.transformation,cloud_open3d_1,cloud_open3d_2

def main(args):
    #good frame 1335705713193323
    #good frame 1335705424378817
    if len(sys.argv) is 3:
        start_at_desired_frame = True
        desired_time_stamp = int(sys.argv[2])
        print("starting at desired time stamp\n")
    else: start_at_desired_frame = False
    
    #initializations
    planes_all = []
    transformation = np.eye(4)
    plane_extractor = PlaneExtracter()
    data_loader = DataLoader()
    point_cloud_operator = PointCloudOperator()
    
    #Get start frame for ICP evaluation
    first_bin = open(sys.argv[1]+str(desired_time_stamp)+".bin", "rb")
    first_cloud = data_loader.load_velodyne_timestep(first_bin)
    first_cloud_open3d=point_cloud_operator.create_open3d_pointcloud(first_cloud[:,:3])
    recent_clouds = []
    trajectory = []
    #Loop through all lidar scans and start with desired scan
    for i,f in enumerate(data_loader.sort_paths(sys.argv[1])):
        #check if current scan is the desired start scan
        if start_at_desired_frame is True and f-desired_time_stamp < 0:
            continue
        start_at_desired_frame = False
        f_bin = open(sys.argv[1]+str(f)+".bin", "rb")
        cloud = data_loader.load_velodyne_timestep(f_bin)
        recent_clouds.append(cloud[:,:3])

        #Skip ICP if for the first frame
        if len(recent_clouds) is not 2:
            continue
        
        #Perform ICP
        transformation_icp,cloud_open3d_1, cloud_open3d_2 = ICP(recent_clouds[0],recent_clouds[1],point_cloud_operator)
        transformation = np.array(transformation_icp) @ transformation
        trajectory.append((f,np.linalg.inv(transformation)))
        del recent_clouds[0]
        
        plane_extractor.getPlaneCorrespondences(cloud_open3d_1, cloud_open3d_2,transformation_icp)

        #TEST AFTER N SECONDS
        if f-desired_time_stamp >  2*10**6:
            point_cloud_operator.evaluate_performance(transformation,first_cloud_open3d,cloud_open3d_2,trajectory)
            break

if __name__ == '__main__':
    sys.exit(main(sys.argv))
