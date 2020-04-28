import SLAMRunner as runner
import SLAMRealisticDataGenerator as sim_generator
import SLAMGtGenerator as gt_generator
import numpy as np
import math

def main():
    np.random.seed(0)

    std_x = np.array([0.3, 0.3, .01]) # x, y, theta
    std_l = np.array([0.01, 0.01, 0.01, 0.005]) # nx, ny, nz, d
    std_p = np.array([0.05, 0.05, 0.001])
    std_noise_l = std_l

    # Choose where to get data from
    # plane_data, world_poses_gt = gt_generator.generate_room_gt()
    move_window = 5
    planes_per_pose = 5
    num_poses = 15
    # calculate number of planes required if you see plane_pose_per frame and
    # see a completely different set of planes every move_window_poses
    num_planes = math.ceil(num_poses * planes_per_pose / move_window)
    plane_data, world_poses_gt = gt_generator.generate_circular_movement_gt(
                                                    num_poses=num_poses, x_movement=1,
                                                    y_movement=1, theta_deg=30,
                                                    num_planes=num_planes)

    planes_gt, midpoints_planes, size_planes = plane_data
    init_pose = world_poses_gt[0]

    data_sim = sim_generator.SimulateData(std_x[0], std_x[-1], std_noise_l[0],
                                          std_noise_l[-1])
    odom_sim, landmark_measurements = data_sim.simulate_random(world_poses_gt, planes_gt,
                                                      move_window=move_window, planes_per_pose=planes_per_pose)
    # odom_sim, landmark_measurements = data_sim.simulate_based_on_closest(world_poses_gt,
    #                                             planes_gt, midpoints_planes, closest_n=5)
    slam_runner = runner.SlamRunner(std_x, std_l, std_p, init_pose)
    slam_runner.add_odometry(odom_sim, world_poses_gt)
    slam_runner.add_plane_measurements(landmark_measurements, planes_gt)
    slam_runner.run()
    print(landmark_measurements.shape)
    # -1 graphs all poses
    #slam_runner.show()
    print("Done with iterations, graphing.")
    slam_runner.show(max_pose_to_graph=-1, show_planes=True,
                     midpoints=midpoints_planes, size=size_planes)



if __name__ == '__main__':
    main()
