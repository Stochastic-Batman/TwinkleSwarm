import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
from tqdm import tqdm


np.random.seed(95)  # ⚡


def visualize_trajectories(trajectories: np.ndarray, output_file: str, title: str = "Drone Swarm", interval: int = 50, show: bool = True):
    os.makedirs('outputs/videos', exist_ok=True)
    output_path = os.path.join('outputs/videos', output_file)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    all_positions = trajectories.reshape(-1, 3)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()
    margin = 2.0
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)
    ax.view_init(elev=20, azim=30)
    scat = ax.scatter([], [], [], c='yellow', marker='o', s=100, edgecolors='orange', linewidths=2.0)
    lines = [ax.plot([], [], [], c='orange', alpha=0.3)[0] for _ in range(trajectories.shape[1])]
    def update(frame):
        positions = trajectories[frame]
        scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        for i, line in enumerate(lines):
            line.set_data(trajectories[:frame+1, i, 0], trajectories[:frame+1, i, 1])
            line.set_3d_properties(trajectories[:frame+1, i, 2])
        ax.set_title(f"{title} - Frame {frame}/{len(trajectories) - 1}")
        return (scat,) + tuple(lines)
    anim = FuncAnimation(fig, update, frames=range(len(trajectories)), interval=interval, blit=False)
    print(f"Saving animation to {output_path}...")
    anim.save(output_path, writer='ffmpeg', fps=5, dpi=150)
    print(f"Animation saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_combined_trajectory(traj_list: list, labels: list, output_file: str, title: str = "Combined Trajectory"):
    os.makedirs('outputs/videos', exist_ok=True)
    output_path = os.path.join('outputs/videos', output_file)
    total_frames = sum(len(t) for t in traj_list)
    combined_traj = np.vstack(traj_list)
    all_positions = combined_traj.reshape(-1, 3)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()
    margin = 2.0
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)
    ax.view_init(elev=20, azim=30)
    scat = ax.scatter([], [], [], c='yellow', marker='o', s=100, edgecolors='orange', linewidths=2.0)
    lines = [ax.plot([], [], [], c='orange', alpha=0.3)[0] for _ in range(combined_traj.shape[1])]
    segment_starts = [0]
    for t in traj_list:
        segment_starts.append(segment_starts[-1] + len(t))

    def update(frame):
        segment_idx = 0
        for i, start in enumerate(segment_starts[:-1]):
            if frame >= start and frame < segment_starts[i + 1]:
                segment_idx = i
                break
        local_frame = frame - segment_starts[segment_idx]
        positions = traj_list[segment_idx][local_frame]
        scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        ax.set_title(f"{title} - {labels[segment_idx]} - Frame {frame}/{total_frames - 1}")
        return scat,
    anim = FuncAnimation(fig, update, frames=range(total_frames), interval=50, blit=False)
    print(f"Saving combined animation to {output_path}...")
    anim.save(output_path, writer='ffmpeg', fps=5, dpi=150)
    print(f"Combined animation saved to {output_path}")
    plt.close(fig)


def export_frames_to_video(trajectories: np.ndarray, output_file: str, fps: int = 20):
    os.makedirs('outputs/videos', exist_ok=True)
    output_path = os.path.join('outputs/videos', output_file)
    frames = []
    all_positions = trajectories.reshape(-1, 3)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()
    margin = 2.0
    print(f"Rendering {len(trajectories)} frames...")
    for frame_idx in tqdm(range(len(trajectories))):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)
        positions = trajectories[frame_idx]
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='yellow', marker='o', s=50, edgecolors='orange', linewidths=1.5)
        ax.set_title(f"Frame {frame_idx}/{len(trajectories)-1}")
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Video saved to {output_path}")