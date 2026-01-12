import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation


np.random.seed(95)  # ⚡


def visualize_trajectories_2d(trajectories: np.ndarray, output_file: str, title: str = "Drone Swarm (Top View)", interval: int = 50, show: bool = True):
    os.makedirs('outputs/videos', exist_ok=True)
    output_path = os.path.join('outputs/videos', output_file)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    all_positions = trajectories.reshape(-1, 3)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    margin = 2.0
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.grid(True, alpha=0.3)
    scat = ax.scatter([], [], c='yellow', marker='o', s=100, edgecolors='orange', linewidths=2.0, zorder=3)
    trail_length = 30
    lines = [ax.plot([], [], c='orange', alpha=0.3, linewidth=1.0, zorder=2)[0] for _ in range(trajectories.shape[1])]
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def update(frame):
        positions = trajectories[frame]
        scat.set_offsets(positions[:, :2])
        start = max(0, frame - trail_length)
        for i, line in enumerate(lines):
            trail = trajectories[start:frame + 1, i]
            line.set_data(trail[:, 0], trail[:, 1])
        frame_text.set_text(f'Frame {frame}/{len(trajectories) - 1}\nTime: {frame * 0.05:.2f}s')
        return (scat,) + tuple(lines) + (frame_text,)

    anim = FuncAnimation(fig, update, frames=range(len(trajectories)), interval=interval, blit=True)
    print(f"Saving 2D animation to {output_path}...")
    anim.save(output_path, writer='ffmpeg', fps=20, dpi=100)
    print(f"2D animation saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_combined_trajectory_2d(traj_list: list, labels: list, output_file: str, title: str = "Combined Trajectory (Top View)"):
    os.makedirs('outputs/videos', exist_ok=True)
    output_path = os.path.join('outputs/videos', output_file)
    total_frames = sum(len(t) for t in traj_list)
    combined_traj = np.vstack(traj_list)
    all_positions = combined_traj.reshape(-1, 3)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    margin = 2.0
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    scat = ax.scatter([], [], c='yellow', marker='o', s=100, edgecolors='orange', linewidths=2.0, zorder=3)
    trail_length = 30
    lines = [ax.plot([], [], c='orange', alpha=0.3, linewidth=1.0, zorder=2)[0] for _ in range(combined_traj.shape[1])]
    segment_starts = [0]
    for t in traj_list:
        segment_starts.append(segment_starts[-1] + len(t))
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def update(frame):
        segment_idx = 0
        for i, start in enumerate(segment_starts[:-1]):
            if frame >= start and frame < segment_starts[i + 1]:
                segment_idx = i
                break
        positions = combined_traj[frame]
        scat.set_offsets(positions[:, :2])
        start = max(0, frame - trail_length)
        for i, line in enumerate(lines):
            trail = combined_traj[start:frame + 1, i]
            line.set_data(trail[:, 0], trail[:, 1])
        ax.set_title(f"{title} - {labels[segment_idx]}")
        frame_text.set_text(f'Frame {frame}/{total_frames - 1}\nPhase: {labels[segment_idx]}')
        return (scat,) + tuple(lines) + (frame_text,)

    anim = FuncAnimation(fig, update, frames=range(total_frames), interval=50, blit=True)
    print(f"Saving combined 2D animation to {output_path}...")
    anim.save(output_path, writer='ffmpeg', fps=20, dpi=100)
    print(f"Combined 2D animation saved to {output_path}")
    plt.close(fig)


def plot_formation_comparison(initial: np.ndarray, final: np.ndarray, targets: np.ndarray, output_file: str, title: str = "Formation Comparison"):
    os.makedirs('outputs/videos', exist_ok=True)
    output_path = os.path.join('outputs/videos', output_file.replace('.mp4', '.png'))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x_min = min(initial[:, 0].min(), final[:, 0].min(), targets[:, 0].min()) - 2
    x_max = max(initial[:, 0].max(), final[:, 0].max(), targets[:, 0].max()) + 2
    y_min = min(initial[:, 1].min(), final[:, 1].min(), targets[:, 1].min()) - 2
    y_max = max(initial[:, 1].max(), final[:, 1].max(), targets[:, 1].max()) + 2
    axes[0].scatter(initial[:, 0], initial[:, 1], c='blue', s=50, alpha=0.6, label='Initial')
    axes[0].set_title('Initial Formation')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].scatter(targets[:, 0], targets[:, 1], c='green', s=50, alpha=0.6, label='Target', marker='x')
    axes[1].set_title('Target Formation')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[2].scatter(final[:, 0], final[:, 1], c='yellow', s=50, edgecolors='orange', linewidths=1.5, alpha=0.8, label='Final')
    axes[2].scatter(targets[:, 0], targets[:, 1], c='green', s=30, alpha=0.3, label='Target', marker='x')
    axes[2].set_title('Final vs Target')
    axes[2].set_xlabel('X (m)')
    axes[2].set_ylabel('Y (m)')
    axes[2].set_xlim(x_min, x_max)
    axes[2].set_ylim(y_min, y_max)
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    error = np.mean(np.linalg.norm(final - targets, axis=1))
    fig.suptitle(f'{title}\nMean Error: {error:.3f} m', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Formation comparison saved to {output_path}")
    plt.close(fig)