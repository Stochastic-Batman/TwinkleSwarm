import numpy as np
from scipy.optimize import linear_sum_assignment
from src.utils import load_trajectory, load_image, get_target_points, get_text_targets, generate_initial_positions
from src.visualize_2d import visualize_trajectories_2d, visualize_combined_trajectory_2d, plot_formation_comparison


def view_problem_1():
    print("\nViewing Problem 1 in 2D...")
    traj = load_trajectory('problem1_static_formation.npy')
    num_drones = traj.shape[1]
    image = load_image('handwritten_name.jpg')
    targets = get_target_points(image, num_drones)
    initial = generate_initial_positions(num_drones, config='cube')
    visualize_trajectories_2d(traj, 'problem1_static_formation_2d.mp4', title='Problem 1: Static Formation (Top View)', show=False)
    final = traj[-1]
    cost_matrix = np.sum((final[:, None] - targets[None]) ** 2, axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mean_error = np.mean(np.linalg.norm(final - targets[col_ind], axis=1))
    plot_formation_comparison(initial, final, targets, 'problem1_comparison', title='Problem 1: Static Formation')
    print(f"Initial config: cube, Final error: {mean_error:.3f} m")


def view_problem_2():
    print("\nViewing Problem 2 in 2D...")
    traj = load_trajectory('problem2_transition.npy')
    num_drones = traj.shape[1]
    targets = get_text_targets("Happy New Year!", num_drones)
    initial = traj[0]
    visualize_trajectories_2d(traj, 'problem2_transition_2d.mp4', title='Problem 2: Transition (Top View)', show=False)
    final = traj[-1]
    cost_matrix = np.sum((final[:, None] - targets[None]) ** 2, axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mean_error = np.mean(np.linalg.norm(final - targets[col_ind], axis=1))
    plot_formation_comparison(initial, final, targets, 'problem2_comparison', title='Problem 2: Transition to Greeting')
    print(f"Final error: {mean_error:.3f} m")


def view_problem_3():
    print("\nViewing Problem 3 in 2D...")
    try:
        traj_trans = load_trajectory('problem3_transition_to_video.npy')
        traj_dyn = load_trajectory('problem3_dynamic_tracking.npy')
        visualize_trajectories_2d(traj_trans, 'problem3_transition_to_video_2d.mp4', title='Problem 3: Transition to Video Shape (Top View)', show=False)
        visualize_trajectories_2d(traj_dyn, 'problem3_dynamic_tracking_2d.mp4', title='Problem 3: Dynamic Tracking (Top View)', show=False)
        print(f"Transition trajectory has {len(traj_trans)} frames, Dynamic has {len(traj_dyn)} frames with {traj_dyn.shape[1]} drones")
    except FileNotFoundError:
        print("Problem 3 trajectories not found yet")


def view_all_combined():
    print("\nViewing combined trajectory in 2D...")
    try:
        traj1 = load_trajectory('problem1_static_formation.npy')
        traj2 = load_trajectory('problem2_transition.npy')
        traj3_trans = load_trajectory('problem3_transition_to_video.npy')
        traj3_dyn = load_trajectory('problem3_dynamic_tracking.npy')
        visualize_combined_trajectory_2d([traj1, traj2, traj3_trans, traj3_dyn], ['Static Formation', 'Transition', 'To Video Shape', 'Dynamic Tracking'], 'combined_all_problems_2d.mp4', title='TwinkleSwarm: Complete Show (Top View)')
    except FileNotFoundError as e:
        print(f"Cannot create combined view: {e}")


if __name__ == '__main__':
    print("=" * 60)
    print("TWINKLESWARM 2D VISUALIZATION")
    print("=" * 60)
    view_problem_1()
    view_problem_2()
    view_problem_3()
    view_all_combined()
    print("\n" + "=" * 60)
    print("2D VISUALIZATION COMPLETE")
    print("=" * 60)
    print("\nOutput files:")
    print("  - outputs/videos/problem1_static_formation_2d.mp4")
    print("  - outputs/videos/problem1_comparison.png")
    print("  - outputs/videos/problem2_transition_2d.mp4")
    print("  - outputs/videos/problem2_comparison.png")
    print("  - outputs/videos/problem3_transition_to_video_2d.mp4")
    print("  - outputs/videos/problem3_dynamic_tracking_2d.mp4")
    print("  - outputs/videos/combined_all_problems_2d.mp4")
    print("")