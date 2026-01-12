import cv2
import os
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


def get_target_points(image: np.ndarray, num_drones: int, invert: bool = False) -> np.ndarray:
    thresh_type = cv2.THRESH_OTSU | (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY)
    _, binary = cv2.threshold(image, 0, 255, thresh_type)
    area = np.sum(binary == 255) / binary.size
    if area > 0.5:
        binary = 255 - binary
    points = np.argwhere(binary == 255)
    points = points[:, [1, 0]]
    if len(points) == 0:
        raise ValueError("No valid points found in image")
    replace = len(points) < num_drones
    indices = np.random.choice(len(points), num_drones, replace=replace)
    targets_2d = points[indices]
    height, width = image.shape
    targets_2d_centered = targets_2d - np.array([width / 2, height / 2])
    targets_3d = np.zeros((num_drones, 3))
    targets_3d[:, 0] = targets_2d_centered[:, 0]
    targets_3d[:, 1] = -targets_2d_centered[:, 1]
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
    return get_target_points(image, num_drones, invert=False)


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