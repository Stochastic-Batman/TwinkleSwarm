import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio


def visualize_trajectories(trajectories: np.ndarray, output_file: str):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter([], [], [])
    def update(frame):
        scat._offsets3d = (trajectories[frame, :, 0], trajectories[frame, :, 1], trajectories[frame, :, 2])
        return scat,
    anim = FuncAnimation(fig, update, frames=range(len(trajectories)), interval=50)
    plt.show()
    anim.save(output_file, writer='imageio')