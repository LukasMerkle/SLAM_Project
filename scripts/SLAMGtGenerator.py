import numpy as np
import grapher as g
import geometry_utils as geom

def generate_room_gt():
    hexagon1_pts, hexagon2_pts = create_hexagons()
    planes, midpoints, size = generate_room_planes(hexagon1_pts, hexagon2_pts)
    planes[planes[:, 3] < 0] *= -1
    planes = geom.normalize_planes(planes)
    world_poses_gt = generate_room_odom()
    plane_data = planes, midpoints, size
    return plane_data, world_poses_gt

def generate_circular_movement_odom(num_poses, x_movement, y_movement, theta_deg):
    odom_list = np.zeros((num_poses, 3))
    odom_list[:, 0] = x_movement
    odom_list[:, 1] = y_movement
    odom_list[:, 2] = np.deg2rad(theta_deg)
    return odom_list

def generate_circular_movement_gt(num_poses, x_movement, y_movement, theta_deg, num_planes):
    gt_odom = generate_circular_movement_odom(num_poses, x_movement, y_movement, theta_deg)

    init_pose = np.array([0, 0, 0])
    world_poses_gt = [init_pose]

    for i,odom in enumerate(gt_odom):
        world_poses_gt.append(geom.compute_world_pose(world_poses_gt[-1], odom))
    world_poses_gt = np.vstack([world_poses_gt])
    midpoints = None
    size = None
    planes_gt = geom.normalize_planes(np.hstack([np.random.randint(2, 10, (num_planes,2)),
                                                 np.zeros((num_planes, 1)),
                                                 np.random.randint(50, 100, (num_planes,1))]))
    plane1 = np.array([0.707, 0.707, 0, 5])
    plane2 = np.array([-0.707, 0.707, 0, 2])
    plane3 = np.array([2, 1, 0, 2])
    plane4 = np.array([3, 4, 0, 2])
    plane5 = np.array([5, 6, 0, 2])
    #planes_gt = geom.normalize_planes(np.vstack([plane1, plane2, plane3, plane4, plane5]))
    plane_data = planes_gt, midpoints, size
    return plane_data, world_poses_gt

def main():
    planes, midpoints, size, world_poses_gt = generate_room_gt()
    g.plot_one_set_planes(planes, midpoints, size, False)
    g.plot_one_trajectory(world_poses_gt)

    # add noise to odometry
    # add noise to measurements
    g.show()

def generate_room_odom():
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

def generate_room_planes(hexagon1_pts, hexagon2_pts):
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
