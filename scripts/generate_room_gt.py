import numpy as np
import grapher as g
import geometry_utils as geom


def generate_gt():
    hexagon1_pts, hexagon2_pts = create_hexagons()
    planes, midpoints, size = generate_planes(hexagon1_pts, hexagon2_pts)
    world_poses_gt = generate_odom()
    return planes, midpoints, size, world_poses_gt

def main():
    planes, midpoints, size, world_poses_gt = generate_gt()
    g.plot_one_set_planes(planes, midpoints, size, False)
    g.plot_one_trajectory(world_poses_gt)
    g.show()

def generate_odom():
    world_pose = np.array([[-4, 0, 60],
                          [-3.7, 1.16, 55],
                          [-2.7, 2.4, 10],
                          [-1.13, 2.5, 0],
                          [0.81, 2.44, -30],
                          [2.42, 1.83, -60],
                          [4.1, 0.01, -80],
                          [3.79, -0.16, -90],
                          [1.96, -1.13, -170],
                          [1.11, -1.58, 180],
                          [-0.89, -1.98, 150],
                          [-3.03, -1.6, 120],
                          [-3.76, -1.07, 110],
                          [-4.1, -0.79, 90],
                          [-4, 0, 60]])
    world_pose[:, 2] = np.deg2rad(world_pose[:, 2])
    return world_pose

def create_hexagons():
    # outer hexagon
    A = [-3, 3]
    B = [2.7, 3.07]
    C = [5, 0]
    D = [1.68, -2.72]
    E = [-3, -2.82]
    F = [-6, 0]

    # inner hexagon
    G = [-2.09, 1.63]
    H = [1.4, 1.66]
    I = [3.03, 0.18]
    J = [1, -1]
    K = [-1.44, -1.12]
    L = [-3, 0]

    hexagon1_pts = np.array([A, B, C, D, E, F])
    hexagon2_pts = np.array([G, H, I, J, K, L])
    return hexagon1_pts, hexagon2_pts

def generate_planes(hexagon1_pts, hexagon2_pts):
    # six points make up a hexagon
    hex_planes1 = []
    hex_planes2 = []
    hex_planes1_size = []
    hex_planes2_size = []
    hex_planes1_mid = []
    hex_planes2_mid = []
    for i in range(6):
        hex_planes1.append(geom.plane_from_line(geom.parametric_line(hexagon1_pts[i % 6],
                                                           hexagon1_pts[(i + 1) % 6])))
        hex_planes2.append(geom.plane_from_line(geom.parametric_line(hexagon2_pts[i % 6],
                                                           hexagon2_pts[(i + 1) % 6])))
        hex_planes1_size.append(geom.length_line(hexagon1_pts[i % 6], hexagon1_pts[(i + 1) % 6]))
        hex_planes2_size.append(geom.length_line(hexagon2_pts[i % 6], hexagon2_pts[(i + 1) % 6]))

        hex_planes1_mid.append(geom.midpoint(hexagon1_pts[i % 6], hexagon1_pts[(i + 1) % 6]))
        hex_planes2_mid.append(geom.midpoint(hexagon2_pts[i % 6], hexagon2_pts[(i + 1) % 6]))

    planes_p = np.vstack([hex_planes1, hex_planes2])
    midpoints = np.vstack([hex_planes1_mid, hex_planes2_mid]).reshape(-1, 2)
    size = np.hstack([hex_planes1_size, hex_planes2_size]).reshape(-1, 1)
    return planes_p, midpoints, size

if __name__ == '__main__':
    main()
