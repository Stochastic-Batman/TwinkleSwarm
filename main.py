import numpy as np
from src.drone_dynamics import compute_trajectories_static, compute_trajectories_transition, compute_trajectories_dynamic
from src.utils import load_image, get_target_points, get_text_targets, generate_initial_positions, save_trajectory, load_trajectory
from src.video_processing import OpticalFlowProcessor, extract_shape_from_video
from src.visualize import visualize_trajectories, visualize_combined_trajectory

np.random.seed(95)  # ⚡


def problem_1_static_formation():
    print("\n" + "=" * 60)
    print("PROBLEM 1: Static Formation on Handwritten Input")
    print("=" * 60)
    num_drones = 150
    image = load_image('handwritten_name.jpg')
    targets = get_target_points(image, num_drones)
    initial_positions = generate_initial_positions(num_drones, config='cube')
    print(f"Computing trajectories for {num_drones} drones...")
    print(f"Initial positions: cube configuration")
    print(f"Target: Handwritten name from image")
    trajectories = compute_trajectories_static(initial_positions, targets, T_final=15.0, dt=0.05)
    save_trajectory(trajectories, 'problem1_static_formation.npy')
    print(f"Trajectory shape: {trajectories.shape}")
    print(f"Final positions reached target: {np.allclose(trajectories[-1], targets, atol=0.5)}")
    visualize_trajectories(trajectories, 'problem1_static_formation.mp4', title='Problem 1: Static Formation', show=False)
    return trajectories


def problem_2_transition_to_greeting():
    print("\n" + "=" * 60)
    print("PROBLEM 2: Transition to New Year Greeting")
    print("=" * 60)
    num_drones = 150
    try:
        traj_problem1 = load_trajectory('problem1_static_formation.npy')
        initial_positions = traj_problem1[-1]
        initial_velocities = np.zeros_like(initial_positions)
        print("Loaded final state from Problem 1")
    except FileNotFoundError:
        print("Problem 1 trajectory not found, running Problem 1 first...")
        traj_problem1 = problem_1_static_formation()
        initial_positions = traj_problem1[-1]
        initial_velocities = np.zeros_like(initial_positions)
    targets = get_text_targets("Happy New Year!", num_drones)
    print(f"Computing transition trajectories for {num_drones} drones...")
    print(f"Initial positions: Final state from Problem 1")
    print(f"Target: 'Happy New Year!' text")
    trajectories = compute_trajectories_transition(initial_positions, initial_velocities, targets, T_final=12.0, dt=0.05)
    save_trajectory(trajectories, 'problem2_transition.npy')
    print(f"Trajectory shape: {trajectories.shape}")
    print(f"Final positions reached target: {np.allclose(trajectories[-1], targets, atol=0.5)}")
    visualize_trajectories(trajectories, 'problem2_transition.mp4', title='Problem 2: Transition to Greeting', show=False)
    return trajectories


def problem_3_dynamic_tracking():
    print("\n" + "=" * 60)
    print("PROBLEM 3: Dynamic Tracking and Shape Preservation")
    print("=" * 60)
    num_drones = 150
    try:
        traj_problem2 = load_trajectory('problem2_transition.npy')
        initial_positions = traj_problem2[-1]
        initial_velocities = np.zeros_like(initial_positions)
        print("Loaded final state from Problem 2")
    except FileNotFoundError:
        print("Problem 2 trajectory not found, running Problem 2 first...")
        traj_problem2 = problem_2_transition_to_greeting()
        initial_positions = traj_problem2[-1]
        initial_velocities = np.zeros_like(initial_positions)
    video_path = 'data/videos/wrecking_ball.mp4'
    print(f"Processing video: {video_path}")
    processor = OpticalFlowProcessor(video_path, scale=0.02, blur_sigma=3.0)
    flows = processor.compute_optical_flow()
    print(f"Computed optical flow for {len(flows)} frames")
    velocity_field_func = processor.get_velocity_field_function(flows)
    video_duration = len(flows) / processor.fps
    T_final = min(video_duration, 10.0)
    dt = 1.0 / processor.fps
    print(f"Computing dynamic tracking trajectories...")
    print(f"Video duration: {video_duration:.2f}s, Simulation duration: {T_final:.2f}s")
    print(f"Frame rate: {processor.fps:.2f} fps")
    trajectories = compute_trajectories_dynamic(initial_positions, initial_velocities, velocity_field_func, T_final, dt)
    save_trajectory(trajectories, 'problem3_dynamic_tracking.npy')
    print(f"Trajectory shape: {trajectories.shape}")
    visualize_trajectories(trajectories, 'problem3_dynamic_tracking.mp4', title='Problem 3: Dynamic Tracking', show=False)
    return trajectories


def run_all_problems():
    print("\n" + "=" * 80)
    print("TWINKLESWARM PROJECT - Illuminated Drone Show Simulation")
    print("=" * 80)
    traj1 = problem_1_static_formation()
    traj2 = problem_2_transition_to_greeting()
    traj3 = problem_3_dynamic_tracking()
    print("\n" + "=" * 60)
    print("Creating combined visualization...")
    print("=" * 60)
    visualize_combined_trajectory([traj1, traj2, traj3], ['Static Formation', 'Transition', 'Dynamic Tracking'], 'combined_all_problems.mp4', title='TwinkleSwarm: Complete Show')
    print("\n" + "=" * 60)
    print("ALL PROBLEMS COMPLETED")
    print("=" * 60)
    print("\nOutput files:")
    print("  - outputs/trajectories/problem1_static_formation.npy")
    print("  - outputs/trajectories/problem2_transition.npy")
    print("  - outputs/trajectories/problem3_dynamic_tracking.npy")
    print("  - outputs/videos/problem1_static_formation.mp4")
    print("  - outputs/videos/problem2_transition.mp4")
    print("  - outputs/videos/problem3_dynamic_tracking.mp4")
    print("  - outputs/videos/combined_all_problems.mp4")
    print("")


if __name__ == '__main__':
    run_all_problems()