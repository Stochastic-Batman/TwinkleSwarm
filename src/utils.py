import os
import cv2
import numpy as np


np.random.seed(95)  # ⚡


def load_image(file_name: str) -> np.ndarray:
    path = os.path.join('data/images', file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image {file_name} not found in data/images")
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def load_video(file_name: str) -> cv2.VideoCapture:
    path = os.path.join('data/videos', file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video {file_name} not found in data/videos")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {file_name}")
    return cap


def get_target_points(image: np.ndarray, num_drones: int) -> np.ndarray:
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contours found in image")
    all_points = []
    for c in contours:
        pts = c.squeeze()
        if pts.ndim == 1:  # Handle single-point contours
            pts = pts.reshape(1, -1)
        if len(pts) > 0:
            all_points.append(pts)
    if len(all_points) == 0:
        raise ValueError("No valid points found in image")
    all_points = np.vstack(all_points)
    if len(all_points) < num_drones:
        indices = np.random.choice(len(all_points), num_drones, replace=True)
    else:
        indices = np.linspace(0, len(all_points) - 1, num_drones, dtype=int)
    targets_2d = all_points[indices]
    targets_3d = np.hstack([targets_2d, np.zeros((num_drones, 1))]).astype(float)
    targets_3d -= np.mean(targets_3d, axis=0)
    max_extent = max(np.ptp(targets_3d[:, 0]), np.ptp(targets_3d[:, 1]))
    if max_extent > 0:
        scale_factor = 30.0 / max_extent
        targets_3d[:, :2] *= scale_factor
    return targets_3d


def get_text_targets(text: str, num_drones: int) -> np.ndarray:
    height, width = 400, 1600
    image = np.zeros((height, width), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 6
    thickness = 15
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, font_scale, 255, thickness, cv2.LINE_AA)
    return get_target_points(image, num_drones)


def generate_initial_positions(num_drones: int, config: str = 'cube') -> np.ndarray:
    np.random.seed(95)
    if config == 'line':
        positions = np.zeros((num_drones, 3))
        positions[:, 0] = np.linspace(-10, 10, num_drones)
    elif config == 'square':
        side = int(np.ceil(np.sqrt(num_drones)))
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) >= num_drones:
                    break
                positions.append([i * 2 - side, j * 2 - side, 0])
        positions = np.array(positions[:num_drones])
    elif config == 'cube':
        side = int(np.ceil(num_drones ** (1/3)))
        positions = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    if len(positions) >= num_drones:
                        break
                    positions.append([i * 2 - side, j * 2 - side, k * 2 - side])
        positions = np.array(positions[:num_drones])
    else:
        raise ValueError(f"Unknown configuration: {config}")
    return positions.astype(float)


def save_trajectory(trajectory: np.ndarray, filename: str):
    os.makedirs('outputs/trajectories', exist_ok=True)
    path = os.path.join('outputs/trajectories', filename)
    np.save(path, trajectory)
    print(f"Trajectory saved to {path}")


def load_trajectory(filename: str) -> np.ndarray:
    path = os.path.join('outputs/trajectories', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory {filename} not found")
    return np.load(path)