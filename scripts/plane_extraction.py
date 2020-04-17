#Usage: python3 velodyne_dataloader.py path_to_velodyne_sync
import sys
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import open3d as o3d

class PlaneExtracter():
    # def __init__(self):

    def extract_planes(self,cloud):
        cloud_open3d= o3d.geometry.PointCloud()
        cloud_open3d.points = o3d.utility.Vector3dVector(cloud[:,:3])

        ground_plane_open3d = cloud_open3d.segment_plane(0.01,3000,1000)

        exluded_ground_points= o3d.geometry.PointCloud()
        exluded_ground_points.points = o3d.utility.Vector3dVector(cloud[~np.isin(np.arange(cloud.shape[0]),ground_plane_open3d[1]),:3])
        
        new_plane_open3d = exluded_ground_points.segment_plane(0.01,10,1000)
        new_plane_open3d_for_vis= o3d.geometry.PointCloud()
        temp_cloud = cloud[~np.isin(np.arange(cloud.shape[0]),ground_plane_open3d[1]),:3]
        new_plane_open3d_for_vis.points = o3d.utility.Vector3dVector(temp_cloud[new_plane_open3d[1],:3])

        self.visualize_poindcloud_and_plane(exluded_ground_points,new_plane_open3d_for_vis)

    def visualize_poindcloud_and_plane(self,pcd,plane):
        pcd.paint_uniform_color([1,1,0])
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
    feature_extractor = PlaneExtracter()
    for i,f in enumerate(os.listdir(sys.argv[1])):
        if i < 1002:
            continue
        f_bin = open(sys.argv[1]+f, "rb")
        # cloud = load_velodyne_timestep(f_bin)
        pcd = o3d.io.read_point_cloud("/home/lmerkle/Downloads/dataset-geometry-only-pc/stimuli/cube_D01_L01.ply")
        cloud = np.asarray(pcd.points)
        plane_extractor = PlaneExtracter()
        plane_extractor.extract_planes(cloud)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
