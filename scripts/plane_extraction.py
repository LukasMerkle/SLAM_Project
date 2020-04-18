#Usage: python3 velodyne_dataloader.py path_to_velodyne_sync
import sys
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
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
        
        for i in range(number_of_planes):
            plane = cloud_open3d.segment_plane(self.min_dist,self.min_number_of_points,self.num_iter)
            
            plane_pointcloud = o3d.geometry.PointCloud()
            plane_pointcloud.points = o3d.utility.Vector3dVector(cloud_numpy[plane[1],:3])
            plane_pointcloud.paint_uniform_color(np.random.rand(3))
            if np.abs(np.dot(plane[0][:3],np.array([0,1,0])))>0.02:
                #check if ground plane by checking dot product with [0,1,0] and exlude it
                pointcloud_list.append(plane_pointcloud)
            cloud_numpy = cloud_numpy[~np.isin(np.arange(cloud_numpy.shape[0]),plane[1]),:3]
            cloud_open3d.points = o3d.utility.Vector3dVector(cloud_numpy)

        o3d.visualization.draw_geometries(pointcloud_list)

    def visualize_poindcloud_and_plane(self,pcd,plane):
        all_points = [pcd]
        all_points.append(plane.paint_uniform_color([1,0,1]))
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

    for i,f in enumerate(sorted_path_list):
        if start_at_desired_frame is True and f-desired_time_stamp < 0:
            continue
        start_at_desired_frame = False
        f_bin = open(sys.argv[1]+str(f)+".bin", "rb")
        cloud = load_velodyne_timestep(f_bin)
        print(f)
        # pcd = o3d.io.read_point_cloud("/home/lmerkle/Downloads/dataset-geometry-only-pc/stimuli/cube_D01_L01.ply")
        # cloud = np.asarray(pcd.points)
        plane_extractor = PlaneExtracter()
        plane_extractor.extract_planes(cloud)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
