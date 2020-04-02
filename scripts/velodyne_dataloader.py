#Usage: python3 velodyne_dataloader.py path_to_velodyne_sync
import sys
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import open3d as o3d

def evaluate_smoothness_scanline(scanline,k_points):
    n_plane_points = 4
    n_edge_points = 2
    smoothness_thresh = 0.1
    smoothness_all = np.zeros((0,4))
    for i in range(k_points,scanline.shape[0]-k_points):
        if not evaluate_patch(scanline[i-k_points:i+k_points+1],k_points):
            continue

        diffs = np.vstack((scanline[i] - scanline[i-k_points:i],
                        scanline[i] - scanline[i+1:i+k_points+1]))
        norm = np.linalg.norm(diffs)
        smoothness = norm/(k_points*2*np.linalg.norm(scanline[i]))
        smoothness_all = np.vstack((smoothness_all,np.array([smoothness,*scanline[i]])))

    sorted_points = smoothness_all[smoothness_all[:,0].argsort()]
    plane_points = sorted_points[:n_plane_points]
    plane_points = plane_points[np.where(plane_points[:,0]<smoothness_thresh)[0]][:,1:]

    edge_points = sorted_points[-n_edge_points:]
    edge_points = edge_points[np.where(edge_points[:,0]>smoothness_thresh)[0]][:,1:]

    return plane_points,edge_points

def evaluate_smoothness_entire_poundcloud(cloud):
    k_points = 5
    k_regions = 6
    all_plane_points = np.zeros((0,3))
    all_edge_points = np.zeros((0,3))
    scan_lines = np.unique(cloud[:,-1])
    for scan_line in scan_lines:
        current_scan_line_points = cloud[np.where(cloud[:,-1] == scan_line)[0],:3]
        plane_points,edge_points = evaluate_smoothness_scanline(current_scan_line_points,k_points)
        all_plane_points = np.vstack((all_plane_points,plane_points))
        all_edge_points = np.vstack((all_edge_points,edge_points))
    return all_plane_points, all_edge_points

def evaluate_patch(patch,feature_index):
    if np.any(np.linalg.norm(patch[:,2]) < np.linalg.norm(patch[feature_index,2])):
        return False
    rand_index = np.random.randint(0,patch.shape[0],3)
    v1 = patch[rand_index[1],:] - patch[rand_index[0],:]
    v2 = patch[rand_index[2],:] - patch[rand_index[0],:]
    cp = np.cross(v1, v2)
    if (np.dot(cp,patch[feature_index,:]) < 0.0002):
        return False
    return True

def convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

def visualize_poindcloud(cloud,color=[1,1,1]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:,:3])
    o3d.visualization.draw_geometries([pcd])

def visualize_edge_and_planes(edge_points,plane_points,cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:,:3])
    pcd.paint_uniform_color([1,1,0])

    pcd_edge = o3d.geometry.PointCloud()
    pcd_edge.points = o3d.utility.Vector3dVector(edge_points)
    pcd_edge.paint_uniform_color([0,0,0])

    pcd_plane = o3d.geometry.PointCloud()
    pcd_plane.points = o3d.utility.Vector3dVector(plane_points)
    pcd_plane.paint_uniform_color([0,0,0.5])

    o3d.visualization.draw_geometries([pcd,pcd_edge,pcd_plane])

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
    for f in os.listdir(sys.argv[1]):
        f_bin = open(sys.argv[1]+f, "rb")
        cloud = load_velodyne_timestep(f_bin)
        all_plane_points, all_edge_points = evaluate_smoothness_entire_poundcloud(cloud)
        visualize_edge_and_planes(all_edge_points, all_plane_points,cloud)


if __name__ == '__main__':
    sys.exit(main(sys.argv))