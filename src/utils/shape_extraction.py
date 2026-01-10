import cv2
import numpy as np


def get_target_points(image: np.ndarray, num_drones: int) -> np.ndarray:
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_points = np.vstack([c.squeeze() for c in contours])
    if len(all_points) < num_drones:
        raise ValueError("Not enough points in image for num_drones")
    indices = np.linspace(0, len(all_points) - 1, num_drones, dtype=int)
    targets_2d = all_points[indices]
    targets_3d = np.hstack([targets_2d, np.zeros((num_drones, 1))])
    targets_3d = targets_3d.astype(float)
    targets_3d -= np.mean(targets_3d, axis=0)
    return targets_3d
