import matplotlib.pyplot as plt
import numpy as np
import pdb;

# Notes to understand file:
# Plotting does not show planes. Acts like matplotlib. Plotting just adds to graph
# If you want to plot and show, run a function that starts with show

def plot_one_trajectory(traj):
    plt.plot(traj[:, 0], traj[:, 1], ls='--')

def show():
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

def plot_one_set_planes(planes_p, midpoints, size, show=False):
    edge_pts1, edge_pts2 = find_line_segment_edge_points(planes_p[:,[0,1,3]], midpoints, size)
    plot_lines(edge_pts1, edge_pts2, "Planes Predicted", 'r')
    if(show):
        plt.show()

def plot_planes(planes_p, planes_gt, midpoints, size=2):
    # line from intesrsection of plane and z= 0 plane is composed of indices
    # 0, 1, 3. ax + by + c = 0
    edge_pts1, edge_pts2 = find_line_segment_edge_points(planes_p[:,[0,1,3]], midpoints, size)
    plot_lines(edge_pts1, edge_pts2, "Planes Predicted", 'r')
    edge_pts1_gt, edge_pts2_gt = find_line_segment_edge_points(planes_gt[:,[0,1,3]], midpoints, size)
    plot_lines(edge_pts1_gt, edge_pts2_gt, "Planes Ground Truth", 'b')

def plot_lines_with_edges(edge_pts1, edge_pts2):
    plot_lines(edge_pts1, edge_pts2, 'r')

'''
    Finds edge points defining a line segment given projective equation vector representing
    line and 2d coordinates of the center of the line and the line segments size
    returns to arrays of each edge of the line. Ex:
    array1 = [[x_edge1_line1], y_edge1_line1], [x_edge1_line2], y_edge1_line2]...]
    array2 = [[x_edge2_line1], y_edge2_line1], [x_edge2_line2], y_edge2_line2]...]
'''
# TODO: pass in actual center point, not just x__center
def find_line_segment_edge_points(lines, center_pts, size):
    A = (lines[:, 0]).reshape(-1, 1)
    B = (lines[:, 1]).reshape(-1, 1)

    theta = np.arctan2(-A, B)
    cos_sin_theta = np.hstack([np.cos(theta), np.sin(theta)])
    edge_pts1 = center_pts + (size / 2) * cos_sin_theta
    edge_pts2 = center_pts - (size / 2) * cos_sin_theta

    return edge_pts1, edge_pts2

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

def show_planes(planes_p, planes_gt, midpoints):
    plot_planes(planes_p, planes_gt, midpoints)
    plt.legend()
    plt.show()

def show_trajectory_and_planes(odom, odom_corrected, odom_ground_truth,
                               planes_p, planes_gt, midpoints):
    plot_trajectories(odom, odom_corrected, odom_ground_truth)
    plot_planes(planes_p, planes_gt, midpoints)
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
    midpoints = np.array([1,2], [2,3], [3,4], [4,5])
    show_trajectory_and_planes(odom, odom_corrected, odom_ground_truth,
                               planes_p, planes_gt, midpoints)


if __name__ == '__main__':
    TEST = False
    if(TEST):
        test_find_line_segment_edge_points()
    main()
