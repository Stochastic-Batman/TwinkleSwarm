import numpy as np
from src.drone_dynamics import compute_trajectories_static, compute_trajectories_transition, params
from src.utils import get_text_targets, generate_initial_positions, save_trajectory
from src.visualize import visualize_trajectories


np.random.seed(95)  # ⚡


def example_0_2d_circle():
    print("\n" + "=" * 60)
    print("EXAMPLE 0: 2D Circle Formation")
    print("=" * 60)
    num_drones = 50
    np.random.seed(95)
    initial_positions = np.random.randn(num_drones, 3) * 5
    initial_positions[:, 2] = 0.0
    angles = np.linspace(0, 2 * np.pi, num_drones, endpoint=False)
    radius = 10.0
    targets_2d = np.zeros((num_drones, 2))
    targets_2d[:, 0] = radius * np.cos(angles)
    targets_2d[:, 1] = radius * np.sin(angles)
    targets = np.hstack([targets_2d, np.zeros((num_drones, 1))])
    print(f"Forming 2D circle pattern with radius {radius}")
    original_k_rep = params['k_rep']
    params['k_rep'] = 0.0
    trajectories = compute_trajectories_static(initial_positions, targets, T_final=25.0, dt=0.1)
    params['k_rep'] = original_k_rep
    save_trajectory(trajectories, 'example0_2d_circle.npy')
    visualize_trajectories(trajectories, 'example0_2d_circle.mp4', title='2D Circle Pattern', show=False)
    print("Example completed")


def example_simple_text():
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Simple Text Formation")
    print("=" * 60)
    num_drones = 80
    text = "HELLO"
    initial_positions = generate_initial_positions(num_drones, config='square')
    targets = get_text_targets(text, num_drones)
    print(f"Forming text '{text}' with {num_drones} drones")
    trajectories = compute_trajectories_static(initial_positions, targets, T_final=20.0, dt=0.1)
    save_trajectory(trajectories, 'example1_simple_text.npy')
    visualize_trajectories(trajectories, 'example1_simple_text.mp4', title=f'Text: {text}', show=False)
    print("Example completed")


def example_multiple_transitions():
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multiple Sequential Transitions")
    print("=" * 60)
    num_drones = 80
    words = ["ONE", "TWO", "THREE"]
    initial_positions = generate_initial_positions(num_drones, config='square')
    all_trajectories = []
    current_positions = initial_positions.copy()
    current_velocities = np.zeros_like(initial_positions)
    for i, word in enumerate(words):
        print(f"Transition {i+1}: '{word}'")
        targets = get_text_targets(word, num_drones)
        trajectories = compute_trajectories_transition(current_positions, current_velocities, targets, T_final=15.0, dt=0.1)
        all_trajectories.append(trajectories)
        current_positions = trajectories[-1]
        if len(trajectories) > 1:
            current_velocities = (trajectories[-1] - trajectories[-2]) / 0.1
        else:
            current_velocities = np.zeros_like(current_positions)
        print(f"  Completed: {len(trajectories)} frames")
    combined_trajectories = np.vstack(all_trajectories)
    save_trajectory(combined_trajectories, 'example2_multiple_transitions.npy')
    visualize_trajectories(combined_trajectories, 'example2_multiple_transitions.mp4', title='Multiple Transitions', show=False)
    print("Example completed")


def example_circular_pattern():
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Circular Pattern")
    print("=" * 60)
    num_drones = 50
    initial_positions = generate_initial_positions(num_drones, config='square')
    angles = np.linspace(0, 2 * np.pi, num_drones, endpoint=False)
    radius = 5.0
    targets = np.zeros((num_drones, 3))
    targets[:, 0] = radius * np.cos(angles)
    targets[:, 1] = radius * np.sin(angles)
    print(f"Forming circular pattern with radius {radius}")
    trajectories = compute_trajectories_static(initial_positions, targets, T_final=20.0, dt=0.1)
    save_trajectory(trajectories, 'example3_circular.npy')
    visualize_trajectories(trajectories, 'example3_circular.mp4', title='Circular Pattern', show=False)
    print("Example completed")


def example_spiral_pattern():
    print("\n" + "=" * 60)
    print("EXAMPLE 4: 3D Spiral Pattern")
    print("=" * 60)
    num_drones = 100
    initial_positions = generate_initial_positions(num_drones, config='square')
    t_vals = np.linspace(0, 4 * np.pi, num_drones)
    radius = 5.0
    targets = np.zeros((num_drones, 3))
    targets[:, 0] = radius * np.cos(t_vals)
    targets[:, 1] = radius * np.sin(t_vals)
    targets[:, 2] = t_vals * 0.5
    print(f"Forming 3D spiral pattern")
    trajectories = compute_trajectories_static(initial_positions, targets, T_final=20.0, dt=0.1)
    save_trajectory(trajectories, 'example4_spiral.npy')
    visualize_trajectories(trajectories, 'example4_spiral.mp4', title='3D Spiral', show=False)
    print("Example completed")


def example_wave_pattern():
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Wave Pattern")
    print("=" * 60)
    num_drones = 80
    initial_positions = generate_initial_positions(num_drones, config='square')
    x_positions = np.linspace(-10, 10, num_drones)
    amplitude = 3.0
    frequency = 0.5
    targets = np.zeros((num_drones, 3))
    targets[:, 0] = x_positions
    targets[:, 1] = amplitude * np.sin(frequency * x_positions)
    print(f"Forming sine wave pattern")
    trajectories = compute_trajectories_static(initial_positions, targets, T_final=20.0, dt=0.1)
    save_trajectory(trajectories, 'example5_wave.npy')
    visualize_trajectories(trajectories, 'example5_wave.mp4', title='Wave Pattern', show=False)
    print("Example completed")


def run_all_examples():
    print("\n" + "=" * 80)
    print("TWINKLESWARM EXAMPLES")
    print("=" * 80)
    example_0_2d_circle()
    example_simple_text()
    example_multiple_transitions()
    example_circular_pattern()
    example_spiral_pattern()
    example_wave_pattern()
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - outputs/trajectories/example*.npy")
    print("  - outputs/videos/example*.mp4")
    print("")


if __name__ == '__main__':
    run_all_examples()