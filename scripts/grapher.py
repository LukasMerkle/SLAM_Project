import matplotlib.pyplot as plt
import numpy as np
import pdb;

# Notes to understand file:
# Plotting does not show planes. Acts like matplotlib. Plotting just adds to graph
# If you want to plot and show, run a function that starts with show

def show_trajectory(traj):
    plt.plot(traj[:, 0], traj[:, 1], ls='--')
    plt.show()


def plot_trajectories(odom, odom_corrected, ground_truth):
    plt.plot(odom[:, 0], odom[:, 1], ls='--', label="Odom ")
    plt.plot(odom_corrected[:, 0], odom_corrected[:, 1], ls='--', label="Odom Corrected")
    plt.plot(ground_truth[:, 0], ground_truth[:, 1], ls='--', label="Odom Ground Truth")

def plot_lines(edge_pts1, edge_pts2, label, c):
    for i, (pt1, pt2) in enumerate(zip(edge_pts1, edge_pts2)):
        # only want to label points once
        if(i == 0):
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], c=c, linewidth=2, label=label)
        else:
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], c=c, linewidth=2)

def plot_planes(planes_p, planes_gt, x_center_vals, size=2):
    # line from intesrsection of plane and z= 0 plane is composed of indices
    # 0, 1, 3. ax + by + c = 0
    edge_pts1, edge_pts2 = find_line_segment_edge_points(planes_p[:,[0,1,3]], x_center_vals, size)
    plot_lines(edge_pts1, edge_pts2, "Planes Predicted", 'r')
    edge_pts1_gt, edge_pts2_gt = find_line_segment_edge_points(planes_gt[:,[0,1,3]], x_center_vals, size)
    plot_lines(edge_pts1_gt, edge_pts2_gt, "Planes Ground Truth", 'b')

'''
    Finds edge points defining a line segment given projective equation vector representing
    line and 2d coordinates of the center of the line and the line segments size
    returns to arrays of each edge of the line. Ex:
    array1 = [[x_edge1_line1], y_edge1_line1], [x_edge1_line2], y_edge1_line2]...]
    array2 = [[x_edge2_line1], y_edge2_line1], [x_edge2_line2], y_edge2_line2]...]
'''
def find_line_segment_edge_points(line, x_center_vals, size):
    x_vals_edge1 = (x_center_vals - size / 2).reshape(-1, 1)
    x_vals_edge2 = (x_center_vals + size / 2).reshape(-1, 1)

    A = (line[:, 0]).reshape(-1, 1)
    B = (line[:, 1]).reshape(-1, 1)
    C = (line[:, 2]).reshape(-1, 1)

    y_vals_edge1_plane = -(C + A * x_vals_edge1) / B
    y_vals_edge2_plane = -(C + A * x_vals_edge2) / B

    return np.hstack((x_vals_edge1, y_vals_edge1_plane)), \
           np.hstack((x_vals_edge2, y_vals_edge2_plane))

def test_find_line_segment_edge_points():
    planes_p = np.array([[2, 3, 10, -2, 0, 1], [4, 3, 20, -4, 0, 1]])
    x_center_vals = np.array([[3,3]])
    size = 2
    edge_pts1, edge_pts2 = find_line_segment_edge_points(planes_p[:,[0,1,3]], x_center_vals, size)
    true_edge_pts1 = np.array([[2,-2/3],[2, -4/3]])
    true_edge_pts2 = np.array([[4, -2],[4, -4]])

    assert(np.all(np.equal(edge_pts1, true_edge_pts1)))
    assert(np.all(np.equal(edge_pts2, true_edge_pts2)))

def show_trajectory(odom, odom_corrected, odom_ground_truth):
    plot_trajectories(odom, odom_corrected, odom_ground_truth)
    plt.legend()
    plt.show()

def show_planes(planes_p, planes_gt, x_center_vals):
    plot_planes(planes_p, planes_gt, x_center_vals)
    plt.legend()
    plt.show()

def show_trajectory_and_planes(odom, odom_corrected, odom_ground_truth,
                               planes_p, planes_gt, x_center_vals):
    plot_trajectories(odom, odom_corrected, odom_ground_truth)
    plot_planes(planes_p, planes_gt, x_center_vals)
    plt.legend()
    plt.show()

def main():
    odom = np.array([[2, -3], [3,-2.7], [3.2,-3.8], [5.5, -5.15], [6.6, -4.6]])
    odom_corrected = np.array([[2,-3], [3, -3.2], [3.7,-4.7], [5, -5.1], [6.2, -4]])
    odom_ground_truth = np.array([[2, -3], [3,-3.7], [3.7,-5], [5, -5], [6, -4]])
    planes_p = np.array([[-1, -0.85, 10, -2, 0, 1], [7, 6,  20, -4, 0, 1],
                         [0, 1, 10, 7, 0, 1], [0, 1,  20, 4, 0, 1]])
    planes_gt = np.array([[-1, -1,  10, -2, 0, 1], [7, 7,  20, -4,  0, 1],
                          [0, 1, 10, 6, 0, 1], [0, 1,  20, 3.5, 0, 1]])
    x_center_vals = np.array([[3, 3, 5, 5]])
    show_trajectory_and_planes(odom, odom_corrected, odom_ground_truth,
                               planes_p, planes_gt, x_center_vals)


if __name__ == '__main__':
    TEST = False
    if(TEST):
        test_find_line_segment_edge_points()
    main()
